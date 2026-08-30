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
        results = collection.query(query_texts=[query], n_results=15)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""
        sources = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            context += f"מקור {i+1}: {doc}\n"
            if meta["url"] not in sources:
                sources.append(meta["url"])

        prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ.

=== איך לענות ===
המידע בסעיף "מידע מאתר ZOOZ" למטה הוא ההקשר שממנו עליך לענות. אם המידע הנתון עונה - ולו באופן חלקי וסביר - על השאלה, ענה עליה בביטחון ובבירור על סמך אותו מידע. רק אם אחרי בדיקה מדוקדקת של כל הקטעים במידע הנתון, אף אחד מהם לא נוגע בכלל לנושא השאלה - אמור בפירוש "אין לי מידע מדויק על כך" והפנה ל-zooz.co.il או ל-info@zooz.co.il. אם ולו קטע אחד מתוך כל הקטעים שסופקו נוגע ברמה כלשהי לנושא, חובה עליך לענות על סמך אותו קטע, גם אם התשובה תהיה חלקית או כללית. אל תסרב לענות רק כי המידע לא מפורט לחלוטין - תשובה כללית סבירה על סמך ההקשר עדיפה על סירוב מיותר.

=== דיוק בפרטים ספציפיים - קריטי ===
כאשר אתה מזכיר עובדה כללית מההקשר (למשל: "יש קשר בין ZOOZ ללקוח מסוים"), אסור לך להוסיף פרטים ספציפיים נוספים - תאריכים, שנים, מספרים, שמות פרויקטים, כמויות - אלא אם הם כתובים מילה-במילה באותו הקשר. אם ההקשר מזכיר לקוח או עובדה בלי תאריך/מספר/שם פרויקט מדויק, ציין את העובדה הכללית בלבד ואל תמציא את הפרט החסר כדי "להשלים" את התשובה. פרט מדויק שגוי (תאריך לא נכון, שם פרויקט מומצא) מזיק הרבה יותר מהיעדר הפרט.

=== חוקים מחייבים ===
1. ענה תמיד בעברית בלבד, גם אם שאלו באנגלית.
2. כשהמשתמש כותב "החברה" או "הכוונה" — הכוונה תמיד ל-ZOOZ.
3. אל תמציא פרטים שלא מופיעים במידע הנתון ולא ניתן להסיק אותם ממנו: שמות שירותים, שמות לקוחות, פרסים או הכרות, נתונים מספריים (כמו גודל צוות, משך פרויקט, מיקום משרדים).
4. פרטי יצירת קשר (טלפון, כתובת פיזית, פקס) - ציין אותם רק אם הם מופיעים במפורש ובאופן מדויק במידע הנתון. אם לא - הפנה רק למייל info@zooz.co.il ולאתר zooz.co.il, ואל תמציא או תנחש מספר טלפון או כתובת.
5. צוות ZOOZ הידוע: ארי מנור הוא המנכ"ל ומייסד החברה. טארק אבו שלבק הוא חלק מהצוות. אל תוסיף עליהם תפקידים, תארים או פרטים שלא מופיעים במידע הנתון.
6. תמצית ולא מלל מיותר: ענה בצורה ממוקדת וברורה, אל תחזור על אותם משפטים כדי להאריך את התשובה.

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
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
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