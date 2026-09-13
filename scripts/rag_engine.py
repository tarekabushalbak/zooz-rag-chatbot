import csv
import os
import re
import time
from datetime import datetime
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
CONTEXT_CHUNKS = 8
CONTACT_URL = "https://www.zooz.co.il/contact.shtml"
MODEL = "openai/gpt-oss-20b"


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


def _item_path(item):
    url = (item.get("metadata") or {}).get("url", "")
    return urlparse(url).path.lower()


def _contains_person_name(item, name="ארי מנור"):
    metadata = item.get("metadata") or {}
    haystack = f"{metadata.get('title', '')} {item.get('document', '')}"
    return name in haystack


def curate_results_for_answer(ranked_results, query):
    """Reduce noisy legacy context for intents where canonical pages exist.

    Retrieval remains broad for recall. Before sending context to the LLM, we make
    canonical company pages dominant for overview/team/contact questions so old
    newsletters and incidental mentions cannot overwhelm a direct answer.
    """
    if not ranked_results:
        return []

    intent = classify_query(query)

    if intent == "contact":
        contact = [item for item in ranked_results if _item_path(item) == "/contact.shtml"]
        return contact[:1] if contact else ranked_results[:3]

    if intent == "team":
        canonical = [item for item in ranked_results if _item_path(item) == "/about_team.shtml"]
        direct = [
            item for item in ranked_results
            if _contains_person_name(item) and _item_path(item) != "/about_team.shtml"
        ]
        selected = canonical + direct
        if not selected:
            selected = ranked_results
        # Keep the person answer focused. Four direct sources are more useful than
        # eight mixed mentions from newsletters, client news or unrelated articles.
        return selected[:4]

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
        return (selected or ranked_results)[:5]

    return ranked_results


def response_guidance(query):
    intent = classify_query(query)
    if intent == "team":
        return (
            "השאלה היא על אדם/צוות. פתח בתפקיד הנוכחי שלו ב-ZOOZ. "
            "ענה ב-2 עד 4 משפטים בלבד. אל תעמיס רשימת לקוחות, פרויקטים או חברות עבר "
            "אלא אם המשתמש ביקש במפורש ביוגרפיה מפורטת."
        )
    if intent == "services_overview":
        return (
            "השאלה היא על שירותי החברה באופן כללי. הסתמך בראש ובראשונה על דף פרופיל החברה/אודות. "
            "סכם את תחומי-העל בלבד: אסטרטגיה, שיווק וחדשנות; וכן ייעוץ ופיתוח ארגוני, "
            "אימון עסקי ופיתוח מנהלים ועובדים. אל תהפוך דוגמאות נקודתיות או כלי CRM לשירות-על."
        )
    if intent == "contact":
        return "השאלה היא על יצירת קשר. תן רק את פרטי הקשר שמופיעים במקור הישיר, בלי הרחבות."
    if intent == "clients":
        return "אם נותנים דוגמאות ללקוחות, ציין רק שמות שמופיעים במפורש במקורות שסופקו."
    return "ענה ישירות לשאלה ואל תוסיף פרטים שאינם נחוצים למענה."


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


def has_current_pricing_evidence(ranked_results):
    """Accept a quoted ZOOZ service price only from an explicit pricing source.

    The legacy site contains many incidental numbers and prices inside old articles,
    newsletters and project examples. Those are never treated as a current service
    price. To avoid false positives, a source must look explicitly like a pricing
    page/title and contain a concrete amount/currency marker.
    """
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
        collection = load_collection()
        ranked = query_collection(
            collection,
            query,
            candidate_k=RETRIEVAL_CANDIDATES,
            top_k=CONTEXT_CHUNKS,
        )

        # Price questions are intentionally conservative. If no explicit current
        # pricing source exists, do not expose incidental amounts from old content.
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
11. בשאלות על מחיר או עלות של שירותי ZOOZ, מחיר שמופיע במאמר, בעלון, בדוגמת לקוח או בפרויקט ישן אינו מחירון של שירותי ZOOZ. אל תציג אותו כמחיר שירות.
12. העדף מידע מדפי אודות, שירותים, צוות, יצירת קשר ועמודי תחום על פני אזכורים מקריים במאמרים ישנים, כאשר הם עונים ישירות על השאלה.
13. אל תוסיף רשימות ארוכות של עובדות צדדיות רק מפני שהן מופיעות במקורות; ענה למה שנשאל.

=== מיקוד מיוחד לשאלה הנוכחית ===
{guidance}

=== מקורות מאתר ZOOZ ===
{context}

=== שאלת המשתמש ===
{query}

=== תשובה ==="""

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
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
