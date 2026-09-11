import csv
import os
import time
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from groq import Groq

try:
    from scripts.retrieval_utils import get_embedding_function, query_collection
except ImportError:
    from retrieval_utils import get_embedding_function, query_collection

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"
RETRIEVAL_CANDIDATES = 40
CONTEXT_CHUNKS = 8


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def log_to_csv(question, answer, duration):
    try:
        log_file = "logs.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "question", "answer", "duration_seconds"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                answer,
                duration,
            ])
    except Exception as exc:
        print(f"Log error: {exc}")


def build_context(ranked_results):
    parts = []
    sources = []

    for index, item in enumerate(ranked_results, start=1):
        document = item["document"]
        metadata = item["metadata"]
        url = metadata.get("url", "")
        title = metadata.get("title", "")

        parts.append(
            f"מקור {index}\n"
            f"כותרת: {title}\n"
            f"כתובת: {url}\n"
            f"תוכן: {document}"
        )
        if url and url not in sources:
            sources.append(url)

    return "\n\n---\n\n".join(parts), sources


def ask_zooz(query):
    start_time = time.time()

    try:
        collection = load_collection()
        ranked = query_collection(
            collection,
            query,
            candidate_k=RETRIEVAL_CANDIDATES,
            top_k=CONTEXT_CHUNKS,
        )
        context, sources = build_context(ranked)

        prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ.

המטרה העליונה שלך היא דיוק. אתה עונה רק על בסיס המידע שמופיע בקטעי המקור שסופקו לך מאתר ZOOZ.

=== כללי מענה מחייבים ===
1. ענה בעברית, בצורה קצרה, ברורה ומקצועית.
2. אל תשתמש בידע כללי, בזיכרון קודם או בהשערות כדי להשלים פרטים שחסרים במקורות.
3. כאשר השאלה מבקשת עובדה ספציפית — למשל מחיר, שם אדם, תפקיד, לקוח, תאריך, מספר, כתובת, טלפון או שם שירות — מותר לציין אותה רק אם היא מופיעה במפורש במקורות.
4. אם המקורות קשורים לנושא אבל אינם מכילים את העובדה המדויקת שנשאלה, אמור: "אין לי מידע מדויק על כך במקורות של ZOOZ שברשותי". לאחר מכן אפשר להפנות ל-zooz.co.il או ל-info@zooz.co.il.
5. אם השאלה מניחה הנחה שאינה נתמכת במקורות, אל תאשר אותה. תקן בעדינות או ציין שאין לכך תמיכה במקורות.
6. אל תמציא שירותים, מוצרים, לקוחות, מחירים, תפקידים, פרויקטים או נתונים מספריים.
7. אם יש כמה מקורות רלוונטיים, חבר ביניהם רק כאשר אין ביניהם סתירה.
8. אם אין במקורות מידע רלוונטי כלל, אמור: "אין לי מידע מדויק על כך" והפנה לאתר ZOOZ או ל-info@zooz.co.il.
9. אל תענה על נושאים שאינם קשורים ל-ZOOZ. במקרה כזה אמור: "אני יכול לעזור רק בנושאים הקשורים ל-ZOOZ."
10. אל תזכיר למשתמש ציוני retrieval, מרחקים וקטוריים או פרטים טכניים פנימיים של המערכת.

=== מקורות מאתר ZOOZ ===
{context}

=== שאלת המשתמש ===
{query}

=== תשובה ==="""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = (
                "לא התקבלה תשובה מהמודל. אנא נסה שוב או פנה ל-info@zooz.co.il"
            )

        duration = round(time.time() - start_time, 2)
        log_to_csv(query, answer, duration)
        return answer, sources

    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"CHATBOT ERROR: {repr(exc)}")
        error_msg = "אירעה שגיאה טכנית. אנא נסה שוב או פנה ל-info@zooz.co.il"
        log_to_csv(query, f"ERROR: {exc}", duration)
        return error_msg, []


if __name__ == "__main__":
    print("ZOOZ Chatbot - type 'exit' to quit")
    while True:
        question = input("Question: ").strip()
        if question == "exit":
            break
        if question:
            answer, sources = ask_zooz(question)
            print(answer)
            print("Sources:")
            for url in sources:
                print(f"  - {url}")
