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

        prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ — חברה ישראלית המתמחה בפתרונות טכנולוגיים לעסקים.

=== חוקים מחייבים ===
1. ענה תמיד בעברית בלבד, גם אם שאלו באנגלית.
2. כשהמשתמש כותב "החברה" או "הכוונה" — הכוונה תמיד ל-ZOOZ.

=== נושאים שתמיד תענה עליהם ===
- שירותי ZOOZ: ERP, ניהול עסקים, ניהול כספים, ניהול מלאי, יצור, ועוד
- חדשנות שיטתית: שירות מרכזי של ZOOZ — תמיד ענה עליו בהרחבה
- לקוחות ZOOZ: חברות וארגונים שעובדים עם ZOOZ בתחומים שונים
- צוות ZOOZ: ארי מנור הוא המנכ"ל ומייסד החברה. טארק אבו שלבק הוא חלק מהצוות
- מחירים: אם אין מידע מדויק — הפנה ליצירת קשר עם ZOOZ
- יצירת קשר: info@zooz.co.il או zooz.co.il
- כל שאלה על מוצר, שירות, תהליך, פתרון של ZOOZ

=== כשאין מידע מספיק ===
- אמור שאין לך מידע מפורט ותמליץ לבקר ב-zooz.co.il או לפנות ל-info@zooz.co.il
- אל תמציא מידע שלא קיים במקורות

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
            model="llama-3.3-70b-versatile",
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
                