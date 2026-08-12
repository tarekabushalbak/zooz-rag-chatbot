import chromadb
import os
import time
import csv
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)

def log_to_csv(question, answer, duration):
    try:
        log_file = "logs.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "question", "answer", "duration_seconds"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, answer, duration])
    except Exception as e:
        print(f"Log error: {e}")

def ask_zooz(query):
    start_time = time.time()
    try:
        collection = load_collection()
        results = collection.query(query_texts=[query], n_results=5)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""
        sources = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            context += f"מקור {i+1}: {doc}\n"
            if meta["url"] not in sources:
                sources.append(meta["url"])

        prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ.

=== חוק עליון - קרא בעיון ===
ענה אך ורק על סמך המידע שמופיע בסעיף "מידע מאתר ZOOZ" למטה. אסור לך בשום אופן להשתמש בידע כללי שלך על חברות, ERP, או תעשיית הטכנולוגיה כדי "למלא חורים" בתשובה. אם המידע הנתון חלקי או לא כולל תשובה ברורה - אמור זאת בפירוש, אל תנחש ואל תמציא.

=== חוקים מחייבים ===
1. ענה תמיד בעברית בלבד, גם אם שאלו באנגלית.
2. כשהמשתמש כותב "החברה" או "הכוונה" — הכוונה תמיד ל-ZOOZ.
3. אסור בהחלט להמציא: שמות שירותים שלא מופיעים במפורש במידע הנתון, שמות לקוחות, פרסים או הכרות, נתונים מספריים (כמו גודל צוות, משך פרויקט, מיקום משרדים) - אלא אם הם כתובים במפורש במידע מאתר ZOOZ למטה.
4. צוות ZOOZ הידוע: ארי מנור הוא המנכ"ל ומייסד החברה. טארק אבו שלבק הוא חלק מהצוות. אל תוסיף עליהם תפקידים, תארים או פרטים שלא מופיעים במידע הנתון.
5. "חדשנות שיטתית" הוא שירות מרכזי של ZOOZ — ענה עליו רק לפי מה שכתוב במידע הנתון, לא לפי הבנה כללית של המושג.
6. תמצית ולא מלל מיותר: ענה בצורה ממוקדת וברורה, אל תחזור על אותם משפטים כדי להאריך את התשובה.

=== כשאין מידע מספיק ===
אם אין במידע הנתון תשובה ברורה וספציפית לשאלה - אמור: "אין לי מידע מדויק על כך" ותמליץ לבקר ב-zooz.co.il או לפנות ל-info@zooz.co.il. אל תנסה "לנחש" תשובה סבירה - זה עדיף על פני מידע שגוי.

=== נושאים שאינם קשורים ל-ZOOZ ===
רק לשאלות שאין להן שום קשר ל-ZOOZ (מזג אוויר, פוליטיקה, ספורט, בידור, אנשים שאינם מ-ZOOZ) — ענה:
"אני יכול לעזור רק בנושאים הקשורים ל-ZOOZ. לשאלות נוספות בקר ב-zooz.co.il"

=== מידע מאתר ZOOZ ===
{context}

=== שאלת המשתמש ===
{query}

=== תשובה ==="""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        answer = response.choices[0].message.content
        duration = round(time.time() - start_time, 2)
        log_to_csv(query, answer, duration)
        return answer, sources

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        print(f"CHATBOT ERROR: {repr(e)}")
        error_msg = f"אירעה שגיאה טכנית. אנא נסה שוב או פנה ל-info@zooz.co.il"
        log_to_csv(query, f"ERROR: {e}", duration)
        return error_msg, []

if __name__ == "__main__":
    print("ZOOZ Chatbot - type 'exit' to quit")
    while True:
        q = input("Question: ").strip()
        if q == "exit":
            break
        if q:
            answer, sources = ask_zooz(q)
            print(answer)
            print("Sources:")
            for url in sources:
                print(f"  - {url}")