import csv
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlparse

import chromadb
from dotenv import load_dotenv
from groq import Groq

try:
    from scripts.retrieval_utils import (
        classify_query,
        get_embedding_function,
        is_pricing_query,
        query_collection,
    )
except ImportError:
    from retrieval_utils import (
        classify_query,
        get_embedding_function,
        is_pricing_query,
        query_collection,
    )

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"
RETRIEVAL_CANDIDATES = 80
CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 1400
CONTACT_URL = "https://www.zooz.co.il/contact.shtml"
CONTACT_EMAIL = "info@zooz.co.il"
CONTACT_PHONE = "09-9585085"
MODEL = "openai/gpt-oss-20b"


@lru_cache(maxsize=1)
def load_collection():
    """Load Chroma and the embedding model once per process."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")
    return Groq(api_key=api_key)


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


def _item_path(item):
    url = (item.get("metadata") or {}).get("url", "")
    return urlparse(url).path.lower()


def _contains_person_name(item, name="ארי מנור"):
    metadata = item.get("metadata") or {}
    haystack = f"{metadata.get('title', '')} {item.get('document', '')}"
    return name in haystack


def curate_results_for_answer(ranked_results, query):
    """Prefer authoritative ZOOZ pages for intents where noisy legacy content exists."""
    if not ranked_results:
        return []

    intent = classify_query(query)

    if intent == "contact":
        contact = [item for item in ranked_results if _item_path(item) == "/contact.shtml"]
        return contact[:1] if contact else ranked_results[:2]

    if intent == "team":
        canonical = [item for item in ranked_results if _item_path(item) == "/about_team.shtml"]
        direct = [
            item for item in ranked_results
            if _contains_person_name(item) and _item_path(item) != "/about_team.shtml"
        ]
        selected = canonical + direct
        return (selected or ranked_results)[:3]

    if intent == "services_overview":
        preferred_paths = [
            "/about_profile.shtml",
            "/about.shtml",
            "/personel.shtml",
        ]
        preferred = []
        for path in preferred_paths:
            preferred.extend(item for item in ranked_results if _item_path(item) == path)
        service_pages = [
            item for item in ranked_results
            if item not in preferred
            and (
                "services" in _item_path(item)
                or "personel" in _item_path(item)
                or "marketing" in _item_path(item)
            )
        ]
        selected = preferred + service_pages
        return (selected or ranked_results)[:4]

    if intent == "clients":
        direct_client_pages = [
            item for item in ranked_results
            if any(marker in _item_path(item) for marker in ["news_clients", "about_clients", "clients"])
        ]
        return (direct_client_pages or ranked_results)[:4]

    if intent in {"triz", "systematic_innovation"}:
        direct = [
            item for item in ranked_results
            if _item_path(item) in {
                "/marketing_article14.shtml",
                "/2-innovation-methods.shtml",
                "/marketing_content_innovation.shtml",
            }
        ]
        return (direct or ranked_results)[:4]

    return ranked_results[:5]


def response_guidance(query):
    intent = classify_query(query)

    if intent == "team":
        return (
            "השאלה היא על אדם/צוות. פתח בתפקיד הנוכחי שלו ב-ZOOZ. "
            "ענה ב-2 עד 4 משפטים בלבד. אל תעמיס רשימת לקוחות, פרויקטים או חברות עבר."
        )

    if intent == "services_overview":
        return (
            "השאלה היא על שירותי החברה באופן כללי. הסתמך בראש ובראשונה על דף פרופיל החברה/אודות. "
            "סכם את תחומי-העל בלבד: אסטרטגיה, שיווק וחדשנות; וכן ייעוץ ופיתוח ארגוני, "
            "אימון עסקי ופיתוח מנהלים ועובדים."
        )

    if intent == "contact":
        return "השאלה היא על יצירת קשר. תן רק את פרטי הקשר הישירים, בלי הרחבות."

    if intent == "clients":
        return (
            "השאלה היא על לקוחות או ארגונים שעבדו עם ZOOZ. השתמש רק בעמודי לקוחות/פרויקטים ישירים. "
            "אל תציג כחלק מלקוחות ZOOZ חברות שמופיעות רק ברקע התעסוקתי של יועץ או עובד."
        )

    if intent == "triz":
        return (
            "השאלה היא על TRIZ. אל תטען ש-ZOOZ פיתחה את TRIZ או את I-TRIZ אלא אם המקור אומר זאת במפורש. "
            "הפרד בין המתודולוגיה עצמה לבין האופן שבו ZOOZ עבדה עם מומחי/פתרונות TRIZ או ייצגה גורם חיצוני."
        )

    if intent == "systematic_innovation":
        return (
            "השאלה היא על שיטות חדשנות. ציין רק שיטות שהמקורות אומרים במפורש ש-ZOOZ משתמשת בהן, מלמדת אותן "
            "או מציעה במסגרת שירותיה. אל תסיק שרשימת מושגים כללית היא רשימת שיטות ש-ZOOZ מלמדת."
        )

    return "ענה ישירות לשאלה ואל תוסיף פרטים שאינם נחוצים למענה."


def _trim_context(text):
    text = (text or "").strip()
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    shortened = text[:MAX_CONTEXT_CHARS]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened + "…"


def build_context(ranked_results):
    parts = []
    sources = []

    for index, item in enumerate(ranked_results, start=1):
        document = _trim_context(item["document"])
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


def has_current_pricing_evidence(ranked_results):
    """Accept a quoted ZOOZ service price only from an explicit pricing source."""
    explicit_price_terms = ("מחיר", "מחירים", "תמחור", "pricing", "price")
    currency_or_amount = re.compile(r"(?:₪|ש\"ח|שח|\$|€|\b\d+[\d,.]*\b)")

    for item in ranked_results:
        metadata = item.get("metadata", {})
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        document = item.get("document", "")
        path = urlparse(url).path.lower()
        title_l = title.lower()
        combined = f"{title} {document}".lower()

        if "/lazooz/" in path or "_article" in path or "/news" in path:
            continue

        explicit_source = (
            "price" in path
            or "pricing" in path
            or any(term in title_l for term in explicit_price_terms)
        )
        if not explicit_source:
            continue

        if any(term in combined for term in explicit_price_terms) and currency_or_amount.search(combined):
            return True

    return False


def ask_zooz(query):
    start_time = time.time()

    try:
        intent = classify_query(query)

        if intent == "out_of_scope":
            answer = "אני יכול לעזור רק בנושאים הקשורים ל-ZOOZ."
            duration = round(time.time() - start_time, 2)
            log_to_csv(query, answer, duration)
            return answer, []

        if intent == "contact" and "כתובת" not in query:
            answer = (
                f"אפשר ליצור קשר עם ZOOZ בטלפון {CONTACT_PHONE} או במייל {CONTACT_EMAIL}. "
                "פרטים נוספים מופיעים בדף יצירת הקשר של החברה."
            )
            duration = round(time.time() - start_time, 2)
            log_to_csv(query, answer, duration)
            return answer, [CONTACT_URL]

        collection = load_collection()
        ranked = query_collection(
            collection,
            query,
            candidate_k=RETRIEVAL_CANDIDATES,
            top_k=CONTEXT_CHUNKS,
        )

        if is_pricing_query(query) and not has_current_pricing_evidence(ranked):
            answer = (
                "אין לי במקורות של ZOOZ מחיר מדויק ועדכני לשירות שנשאל. "
                "לקבלת הצעת מחיר מומלץ לפנות ל-ZOOZ דרך info@zooz.co.il או דרך אתר החברה."
            )
            duration = round(time.time() - start_time, 2)
            log_to_csv(query, answer, duration)
            return answer, [CONTACT_URL]

        curated_ranked = curate_results_for_answer(ranked, query)
        context, sources = build_context(curated_ranked)
        guidance = response_guidance(query)

        prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ.

ענה רק על בסיס המקורות שסופקו מאתר ZOOZ. המטרה העליונה היא דיוק.

כללים:
1. ענה בעברית, קצר וברור.
2. אל תשתמש בידע חיצוני ואל תשלים פרטים חסרים מהשערה.
3. עובדה ספציפית כמו מחיר, שם, תפקיד, לקוח, תאריך, מספר או שירות מותרת רק אם היא מופיעה במפורש במקורות.
4. אם העובדה המדויקת לא מופיעה, אמור שאין לך מידע מדויק עליה במקורות של ZOOZ.
5. אל תאשר הנחה לא נתמכת ואל תמציא שירותים, לקוחות, מחירים או נתונים.
6. העדף דפי אודות, שירותים, צוות, קשר ועמודי תחום על פני אזכורים מקריים במאמרים ישנים.
7. אל תענה על נושאים שאינם קשורים ל-ZOOZ.
8. אל תזכיר ציוני retrieval או פרטים פנימיים של המערכת.
9. מחיר שמופיע במאמר/עלון/דוגמה ישנה אינו מחירון שירותי ZOOZ.
10. ענה רק למה שנשאל ואל תעמיס פרטים צדדיים.

מיקוד לשאלה:
{guidance}

מקורות:
{context}

שאלה:
{query}

תשובה:"""

        client = get_groq_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            reasoning_effort="low",
            include_reasoning=False,
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Groq returned empty answer content")

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