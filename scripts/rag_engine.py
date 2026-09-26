import csv
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from urllib.parse import unquote, urljoin, urlparse, urldefrag

import chromadb
import requests
from bs4 import BeautifulSoup
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
FALLBACK_MODEL = "openai/gpt-oss-120b"
MAX_COMPLETION_TOKENS = 650
EXACT_PAGE_MAX_CHARS = 14000
ZOOZ_HOSTS = {"zooz.co.il", "www.zooz.co.il"}
REFERENCE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:zooz\.co\.il|tinyurl\.com)/[^\s<>\"']+",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_collection():
    """Load Chroma and the embedding model once per process."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


@lru_cache(maxsize=1)
def load_read_collection():
    """Open the existing collection for document/metadata reads only.

    No embedding function is attached here, so exact-URL lookups and source labels
    do not load the sentence-transformer model into memory.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")
    # Disable SDK-level retries. A 429 should return immediately so our own
    # primary->fallback logic can run before Gunicorn's request timeout.
    return Groq(api_key=api_key, max_retries=0, timeout=20.0)


def _clean_inline_text(text):
    return " ".join((text or "").split())


def _normalize_zooz_url(url):
    raw = (url or "").strip().rstrip(".,;:!?)]}>'\"״׳")
    if not raw:
        return ""

    raw, _ = urldefrag(raw)
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in ZOOZ_HOSTS:
        return ""

    return parsed._replace(
        scheme="https",
        netloc="www.zooz.co.il",
        query="",
        fragment="",
    ).geturl()


def _resolve_reference_url(url):
    raw = (url or "").strip().rstrip(".,;:!?)]}>'\"״׳")
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()

    if host in ZOOZ_HOSTS:
        return _normalize_zooz_url(raw)

    # Resolve TinyURL safely one hop only. We never follow an arbitrary redirect:
    # the Location header must already point directly to zooz.co.il.
    if host == "tinyurl.com" and parsed.scheme in {"http", "https"}:
        try:
            response = requests.get(
                raw,
                timeout=4,
                allow_redirects=False,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ZOOZ-RAG/3.0)"},
            )
            try:
                location = response.headers.get("location", "")
            finally:
                response.close()
            if location:
                target = urljoin(raw, location)
                return _normalize_zooz_url(target)
        except Exception as exc:
            print(f"INFO: could not resolve TinyURL reference: {exc}")
    return ""


def extract_zooz_reference_urls(text):
    resolved = []
    seen = set()
    for match in REFERENCE_URL_RE.findall(text or ""):
        url = _resolve_reference_url(match)
        if url and url not in seen:
            seen.add(url)
            resolved.append(url)
    return resolved


def _extract_live_page_content(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()

    blocks = []
    seen = set()
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if tag.name == "td" and tag.find(["h1", "h2", "h3", "h4", "p", "li"]):
            continue
        text = _clean_inline_text(tag.get_text(" ", strip=True))
        if len(text) < 12 or text in seen:
            continue
        seen.add(text)
        blocks.append(text)

    if not blocks:
        text = _clean_inline_text(soup.get_text(" ", strip=True))
    else:
        text = "\n".join(blocks)

    if len(text) > EXACT_PAGE_MAX_CHARS:
        text = text[:EXACT_PAGE_MAX_CHARS]
        if " " in text:
            text = text.rsplit(" ", 1)[0]
        text += "…"
    return text


@lru_cache(maxsize=160)
def _fetch_live_zooz_page(url):
    normalized = _normalize_zooz_url(url)
    if not normalized:
        return None

    try:
        response = requests.get(
            normalized,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZOOZ-RAG/3.0)"},
        )
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        if content_type and "html" not in content_type:
            return None

        if not response.encoding or response.encoding.lower() in {"iso-8859-1", "ascii"}:
            response.encoding = response.apparent_encoding or "utf-8"

        soup = BeautifulSoup(response.text, "lxml")
        h1_tag = soup.find("h1")
        h1 = _clean_inline_text(h1_tag.get_text(" ", strip=True)) if h1_tag else ""
        title = _clean_inline_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        content = _extract_live_page_content(soup)
        if not content:
            return None

        return {
            "url": normalized,
            "h1": h1,
            "title": title,
            "content": content,
            "origin": "live",
        }
    except Exception as exc:
        print(f"INFO: direct ZOOZ page fetch failed for {normalized}: {exc}")
        return None


def _load_indexed_zooz_page(url):
    normalized = _normalize_zooz_url(url)
    if not normalized:
        return None

    try:
        collection = load_read_collection()
        result = collection.get(
            where={"url": normalized},
            include=["documents", "metadatas"],
        )
        raw_documents = result.get("documents") or []
        raw_metadatas = result.get("metadatas") or []
        pairs = [
            (doc, meta or {})
            for doc, meta in zip(raw_documents, raw_metadatas)
            if doc
        ]
        if not pairs:
            return None

        # Chroma does not guarantee get() order, so reconstruct page order from
        # the chunk_index saved during indexing.
        pairs.sort(key=lambda pair: int(pair[1].get("chunk_index", 0) or 0))
        documents = [doc for doc, _ in pairs]
        metadatas = [meta for _, meta in pairs]

        content = "\n\n".join(documents)
        if len(content) > EXACT_PAGE_MAX_CHARS:
            content = content[:EXACT_PAGE_MAX_CHARS]
            if " " in content:
                content = content.rsplit(" ", 1)[0]
            content += "…"

        title = ""
        for metadata in metadatas:
            title = _clean_inline_text(metadata.get("title", ""))
            if title:
                break

        # The crawler preserves semantic headings inside each chunk, but the old
        # Chroma metadata stores only <title>. Infer the first page heading from
        # chunk 0 so the UI can show a meaningful H1-style source label, as Ari
        # requested, instead of "מקור 1".
        h1 = ""
        first_lines = [
            _clean_inline_text(line)
            for line in documents[0].splitlines()
            if _clean_inline_text(line)
        ]
        if first_lines and title and first_lines[0] == title:
            first_lines = first_lines[1:]
        for candidate in first_lines[:4]:
            if candidate == title:
                continue
            if 3 <= len(candidate) <= 140:
                h1 = candidate
                break

        return {
            "url": normalized,
            "h1": h1,
            "title": title,
            "content": content,
            "origin": "index",
        }
    except Exception as exc:
        print(f"INFO: exact URL lookup in ChromaDB failed for {normalized}: {exc}")
        return None


def _fallback_source_label(url):
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()

    if "linkedin.com" in host:
        if "ari-manor" in parsed.path.lower():
            return "Ari Manor – LinkedIn"
        return "LinkedIn"

    basename = unquote((parsed.path or "").rstrip("/").split("/")[-1])
    if basename:
        if basename.lower().endswith((".shtml", ".html", ".htm")):
            basename = basename.rsplit(".", 1)[0]
        return basename.replace("_", " ").replace("-", " ").strip()

    return host or "מקור"


def source_label_for_url(url):
    normalized = _normalize_zooz_url(url)
    if normalized:
        # Prefer the local crawl for source labels. It is fast, stable, and avoids
        # making one external HTTP request per displayed source.
        indexed_page = _load_indexed_zooz_page(normalized)
        if indexed_page:
            label = indexed_page.get("h1") or indexed_page.get("title")
            if label:
                return _clean_inline_text(label)[:120]

        page = _fetch_live_zooz_page(normalized)
        if page:
            # Ari asked specifically for the page H1. Fall back to <title> only
            # on legacy pages that do not expose an H1.
            label = page.get("h1") or page.get("title")
            if label:
                return _clean_inline_text(label)[:120]

    return _fallback_source_label(url)[:120]


def _reference_question_without_urls(question):
    clean = REFERENCE_URL_RE.sub(" ", question or "")
    clean = re.sub(r"\s+", " ", clean).strip(" ,;:-")
    return clean


def _is_generic_link_instruction(text):
    q = _clean_inline_text(text).lower()
    if not q:
        return True
    generic = (
        "התשובה נמצאת כאן",
        "תושבה נמצאת כאן",
        "ענה שוב",
        "עני שוב",
        "בקישור הזה",
        "במאמר הזה",
        "לפי הקישור הזה",
        "לפי המאמר הזה",
        "תסביר לי יותר",
        "תסביר יותר",
        "תפרט",
        "תפרט יותר",
        "תרחיב",
        "הרחב",
        "כאן",
    )
    return len(q) < 70 and any(term in q for term in generic)


def _is_anaphoric_page_followup(text):
    q = _clean_inline_text(text).lower()
    return any(term in q for term in (
        "ביניהם",
        "ביניהן",
        "אותם",
        "אותן",
        "אלה",
        "אלו",
        "הכלים האלה",
        "הכלים האלו",
    ))


def _page_query_terms(text):
    tokens = re.findall(r"[A-Za-z0-9\u0590-\u05FF]+", (text or "").lower())
    stopwords = {
        "לפי", "הזה", "הזאת", "מה", "איך", "למה", "של", "על", "עם", "את",
        "בין", "ומה", "תסביר", "תפרט", "יותר", "המאמר", "הקישור", "דף",
        "this", "the", "and", "from", "with", "page", "article",
    }
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


def _select_exact_page_excerpt(content, query, max_chars=6200):
    """Select the most relevant article blocks without loading embeddings."""
    text = (content or "").strip()
    if len(text) <= max_chars:
        return text

    blocks = [block.strip() for block in re.split(r"\n{1,2}", text) if block.strip()]
    terms = _page_query_terms(query)
    if not blocks:
        return text[:max_chars]

    scored = []
    for index, block in enumerate(blocks):
        lowered = block.lower()
        score = sum(1 for term in terms if term in lowered)
        if score:
            scored.append((score, index))

    if not scored:
        return text[:max_chars]

    selected_indices = set()
    for _, index in sorted(scored, key=lambda x: (-x[0], x[1]))[:12]:
        selected_indices.update({
            max(0, index - 1),
            index,
            min(len(blocks) - 1, index + 1),
        })

    pieces = []
    used = 0
    for index in sorted(selected_indices):
        block = blocks[index]
        addition = len(block) + (2 if pieces else 0)
        if used + addition > max_chars:
            remaining = max_chars - used
            if remaining > 120:
                pieces.append(block[:remaining].rsplit(" ", 1)[0] + "…")
            break
        pieces.append(block)
        used += addition

    return "\n\n".join(pieces) if pieces else text[:max_chars]


def answer_from_zooz_page_reference(question, previous_question=""):
    """Answer from the exact ZOOZ page the user referenced.

    This is intentionally separate from semantic retrieval: a URL is an identifier,
    not a semantic query. If the page is not present in the packaged Chroma snapshot,
    we read the public ZOOZ page directly and ground the answer in that page only.
    """
    urls = extract_zooz_reference_urls(question)
    if not urls:
        return None

    pages = []
    for url in urls[:2]:
        # Prefer the local crawl/index. It is the same corpus used by RAG and
        # avoids the 403 that zooz.co.il currently returns to Render.
        page = _load_indexed_zooz_page(url) or _fetch_live_zooz_page(url)
        if page:
            pages.append(page)

    if not pages:
        return None

    requested = _reference_question_without_urls(question)
    previous = _reference_question_without_urls(previous_question)

    # A turn containing only the URL means: answer the substantive question
    # that immediately led to the user supplying this source.
    if not requested:
        if previous:
            requested = previous
        else:
            requested = "סכם את המידע המרכזי בדף והסבר מה ניתן ללמוד ממנו."

    # Resolve pronouns before generic-link handling. Otherwise a question such as
    # "ומה ההבדל ביניהם לפי המאמר הזה?" gets mistaken for "summarize the article".
    elif _is_anaphoric_page_followup(requested) and previous:
        requested = (
            f"השאלה הקודמת הייתה: {previous}\n"
            f"שאלת ההמשך היא: {requested}\n"
            "המילים 'ביניהם/ביניהן/אלה/אלו' מתייחסות לנושא שהוגדר בשאלה הקודמת. "
            "ענה על שאלת ההמשך עצמה, ולא על סיכום כללי של הדף."
        )

    elif _is_generic_link_instruction(requested):
        if previous and not _is_generic_link_instruction(previous):
            requested = (
                f"הרחב את התשובה לשאלה הקודמת, תוך הסתמכות על הדף בלבד: {previous}"
            )
        else:
            requested = "סכם את המידע המרכזי בדף והסבר מה ניתן ללמוד ממנו."

    page_context = []
    for index, page in enumerate(pages, start=1):
        heading = page.get("h1") or page.get("title") or source_label_for_url(page["url"])
        excerpt = _select_exact_page_excerpt(page["content"], requested)
        page_context.append(
            f"דף {index}\n"
            f"כותרת: {heading}\n"
            f"כתובת: {page['url']}\n"
            f"תוכן רלוונטי מהדף:\n{excerpt}"
        )

    prompt = f"""אתה העוזר הווירטואלי של חברת ZOOZ.

המשתמש הפנה במפורש לדף מסוים באתר ZOOZ. ענה על בסיס תוכן הדף המצורף בלבד.
ענה קודם כל על השאלה המדויקת שנשאלה; אל תחליף אותה בסיכום כללי של המאמר.
אל תגיד שהמודל "לא אומן" על הדף. אם התשובה נמצאת בדף, חלץ אותה במדויק והסבר אותה.
אל תערבב ידע מדפים אחרים ואל תשלים מידע שאינו נתמך בדף.
אל תוסיף שמות של שיטות, כלי-משנה, דוגמאות, אחוזים או תיאורים שאינם מופיעים במפורש בתוכן הדף שסופק.
התייחס לרשימות ניווט, "למידע נוסף", "מידע משלים", קישורים למאמרים/סדנאות ותוכן צדדי כאל ניווט בלבד — לא כהוכחה לכך שהמאמר עצמו מסביר או משווה את אותם נושאים.
אם השאלה מבקשת השוואה, הבדל או מתי להשתמש בכלי מסוים, ענה רק אם גוף המאמר עצמו נותן מידע מספיק להשוואה. אל תבנה השוואה רק משום ששמות של כלים מופיעים ברשימת קישורים.
אם אין בגוף המאמר השוואה מפורשת או מידע מספיק, פתח במשפט ברור: "המאמר אינו מספק השוואה מלאה בנושא הזה", ואז ציין בקצרה רק מה הוא כן אומר.
אם המשתמש משתמש בכינויי המשך כמו "ביניהם" או "הכלים האלה", פרש אותם רק לפי ההקשר שניתן בשאלת ההמשך.
אל תחליף בקשת "הבדל/השוואה" בסיכום כללי של המאמר.
ענה בעברית ברורה, עניינית ומלאה. אם הדף אינו מספק את הפרט המבוקש, אמור זאת במפורש.

שאלת המשתמש:
{requested}

הדף/ים:
{chr(10).join(page_context)}

תשובה:"""

    client = get_groq_client()
    response = _create_completion(client, prompt, max_tokens=450)
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("Groq returned empty answer content for exact-page request")

    return answer, [page["url"] for page in pages]


def log_to_csv(question, answer, duration):
    """Keep the lightweight local log as a best-effort secondary log."""
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


def _normalized_question(query):
    return " ".join((query or "").lower().split())


def _is_mixed_pricing_question(query):
    q = _normalized_question(query)
    pricing_terms = ("כמה עולה", "כמה עולים", "מה המחיר", "מחיר", "מחירים", "עלות", "תמחור")
    if not any(term in q for term in pricing_terms):
        return False

    broader_markers = (
        "לוח זמנים", "כמה זמן", "מי יעביר", "מי ינחה", "מי מעביר",
        "תוצאות מובטחות", "תוצאה מובטחת", "מה הערך", "איזה חלק",
        "אילו חלקים", "איזה מידע", "האם עדיין", "גם אם", "אם אין",
        "אם לא", "ובנוסף", "וגם",
    )
    return any(marker in q for marker in broader_markers) or q.count("?") > 1 or q.count(",") >= 2


def _strip_pricing_terms(query):
    cleaned = re.sub(
        r"כמה\s+עולה|כמה\s+עולים|מה\s+המחיר|מחיר(?:ים)?|עלות|תמחור",
        " ",
        query or "",
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split()).strip(" ,;:-")
    return cleaned or query


def _is_company_inference_question(query):
    q = _normalized_question(query)
    inference_markers = (
        "מה אפשר להסיק", "מה ניתן להסיק", "מה אפשר ללמוד", "מה ניתן ללמוד",
        "מסקנה אחת", "מסקנה חזקה", "איזו מסקנה", "איזה יתרון אפשר",
    )
    if not any(marker in q for marker in inference_markers):
        return False

    specific_terms = (
        "לקוחות", "לקוח", "triz", "ארי מנור", "טלפון", "אימייל",
        "מייל", "כתובת", "מחיר", "עלות",
    )
    return not any(term in q for term in specific_terms)


def _is_inventive_tools_question(query):
    q = _normalized_question(query)
    return any(term in q for term in (
        "כלי החשיבה ההמצאתית",
        "כלי חשיבה המצאתית",
        "כלי-חשיבה המצאתית",
        "שישה כלי חשיבה",
        "ששת כלי החשיבה",
    ))


def _is_multiplication_tool_question(query):
    q = _normalized_question(query)
    return any(term in q for term in (
        "מה זה הכפלה",
        "מהי הכפלה",
        "איך מבצעים הכפלה",
        "איך עושים הכפלה",
        "כלי הכפלה",
        "הכפל מרכיב",
        "הכפלה?",
    )) or q.strip(' ?!.׳"״') == "הכפלה" or (_is_inventive_tools_question(query) and "הכפלה" in q)


def _is_geographic_clients_question(query):
    q = _normalized_question(query)
    geography_terms = (
        "באיזה ארצות", "באילו ארצות", "איזה ארצות", "באיזה מדינות", "באילו מדינות",
        "איזה מדינות", "בחו\"ל", "בחו״ל", "בעולם", "מחוץ לישראל", "הונג קונג",
    )
    work_terms = ("עבד", "עובד", "לקוחות", "לקוח", "פרויקט", "פעילות")
    return any(term in q for term in geography_terms) and any(term in q for term in work_terms)


def _is_positioning_question(query):
    q = _normalized_question(query)
    return "מיצוב" in q


def _merge_ranked_results(*groups, limit=12):
    merged = []
    seen = set()
    for group in groups:
        for item in group or []:
            metadata = item.get("metadata") or {}
            key = (metadata.get("url", ""), item.get("document", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _unsupported_fact_response(query):
    q = _normalized_question(query)
    risky_patterns = (
        "erp", "שירות התשלומים", "שירותי תשלומים", "פרס נובל", "כמה עובדים",
        "מספר העובדים", "הכנסות השנתיות", "מה ההכנסות", "כמה סניפים",
        "מספר סניפים", "תמציא לי", "לא מופיעה באתר", "לא מופיע באתר",
    )
    if any(pattern in q for pattern in risky_patterns):
        return (
            "אין לי מידע במקורות של ZOOZ שמאשר את הפרט או ההנחה שבשאלה, "
            "ולכן לא אמציא תשובה."
        )
    return ""


def _is_rate_limit_error(exc):
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status == 429 or "rate limit" in text or "rate_limit_exceeded" in text


def _create_completion(client, prompt, max_tokens=MAX_COMPLETION_TOKENS):
    """Use the main model and fail over to a second Groq model on a 429 limit."""
    common = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    try:
        return client.chat.completions.create(
            model=MODEL,
            reasoning_effort="low",
            include_reasoning=False,
            **common,
        )
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        # A 429 from the primary model is expected when its free-tier bucket is full.
        # Keep the log informational and try the separately-limited fallback model.
        print("INFO: primary Groq model temporarily rate-limited; using fallback model.")
        return client.chat.completions.create(
            model=FALLBACK_MODEL,
            **common,
        )


def curate_results_for_answer(ranked_results, query):
    """Prefer authoritative/direct ZOOZ sources for sensitive or noisy intents."""
    if not ranked_results:
        return []

    intent = classify_query(query)

    if _is_inventive_tools_question(query) or _is_multiplication_tool_question(query):
        priority_markers = (
            "/marketing_article3.shtml",
            "/marketing_article3a.shtml",
            "/marketing_article3b.shtml",
            "/marketing_article3c.shtml",
            "/2-innovation-tools.shtml",
            "/2-product-innovation.shtml",
            "/personel_article24.shtml",
            "/workshops_marketing.shtml",
            "/zooz-workshops-innovation.pdf",
        )
        direct = [
            item for item in ranked_results
            if any(marker in _item_path(item) for marker in priority_markers)
        ]
        return (direct or ranked_results)[:5]

    if _is_geographic_clients_question(query):
        direct = [
            item for item in ranked_results
            if any(marker in _item_path(item) for marker in ("news_clients", "about_clients", "clients"))
        ]
        return (direct or ranked_results)[:5]

    if _is_positioning_question(query):
        direct = [
            item for item in ranked_results
            if "marketing" in _item_path(item) or "lazooz" in _item_path(item)
        ]
        return (direct or ranked_results)[:5]

    if _is_company_inference_question(query):
        preferred_paths = ["/about_profile.shtml", "/about.shtml", "/personel.shtml"]
        preferred = []
        for path in preferred_paths:
            preferred.extend(item for item in ranked_results if _item_path(item) == path)

        direct_pages = []
        for item in ranked_results:
            if item in preferred:
                continue
            path = _item_path(item)
            if "/lazooz/" in path or "_article" in path or "/news" in path:
                continue
            if any(marker in path for marker in ("services", "innovation", "marketing", "personel", "clients", "about")):
                direct_pages.append(item)
        selected = _merge_ranked_results(preferred, direct_pages, limit=5)
        return selected or ranked_results[:5]

    if intent == "contact":
        contact = [item for item in ranked_results if _item_path(item) == "/contact.shtml"]
        return contact[:1] if contact else ranked_results[:2]

    if intent == "team":
        preferred_paths = (
            "/about_team.shtml",
            "/marketing_article13.shtml",
            "/lazooz/lazooz89.html",
            "/marketing-training.pdf",
        )
        preferred = []
        for path in preferred_paths:
            preferred.extend(item for item in ranked_results if _item_path(item) == path)
        direct = [
            item for item in ranked_results
            if _contains_person_name(item) and item not in preferred
        ]
        return (_merge_ranked_results(preferred, direct, limit=4) or ranked_results[:4])

    if intent == "services_overview":
        preferred_paths = ["/about_profile.shtml", "/about.shtml", "/personel.shtml"]
        preferred = []
        for path in preferred_paths:
            preferred.extend(item for item in ranked_results if _item_path(item) == path)
        service_pages = [
            item for item in ranked_results
            if item not in preferred and any(marker in _item_path(item) for marker in ("services", "personel", "marketing"))
        ]
        return (_merge_ranked_results(preferred, service_pages, limit=5) or ranked_results[:5])

    if intent == "clients":
        direct = [
            item for item in ranked_results
            if any(marker in _item_path(item) for marker in ("news_clients", "about_clients", "clients"))
        ]
        return (direct or ranked_results)[:5]

    if intent in {"triz", "systematic_innovation"}:
        direct = [
            item for item in ranked_results
            if _item_path(item) in {
                "/marketing_article14.shtml",
                "/2-innovation-methods.shtml",
                "/marketing_content_innovation.shtml",
                "/2-innovation-tools.shtml",
            }
        ]
        return (direct or ranked_results)[:5]

    return ranked_results[:5]


def response_guidance(query):
    intent = classify_query(query)

    if _is_mixed_pricing_question(query):
        return (
            "השאלה כוללת כמה רכיבים. ענה לכל רכיב בנפרד ובקיצור. אם אין במקורות מחיר, לוח זמנים, "
            "תוצאה מובטחת או פרט אחר, ציין שחסר מידע רק לגבי אותו רכיב והמשך לענות על השאר."
        )

    if _is_inventive_tools_question(query):
        return (
            "השאלה היא על כלי החשיבה ההמצאתית. תן תשובה שימושית ולא רק שמות: פתח בהסבר קצר על SIT, "
            "לאחר מכן הצג את ששת הכלים שמופיעים במקורות — הכפלה, חלוקה, החסרה, איחוד, הוספת מימד "
            "והתאמה לסביבה — ובמשפט קצר הסבר מה עושה כל כלי. אל תחליף כלי אחד באחר ואל תחזור על אותו כלי."
        )

    if _is_multiplication_tool_question(query):
        return (
            "השאלה היא על כלי החשיבה 'הכפלה'. הסבר במפורש שהפעולה היא לבחור מרכיב קיים במוצר או בשירות, "
            "להכפיל אותו (אפשר גם עם שינוי), ואז לבחון איזו תועלת חדשה נוצרת. תן דוגמה קונקרטית מן המקורות "
            "והסבר מה התועלת בדוגמה. אל תפרש 'הכפלה' כהכפלת טקסט או מסמך."
        )

    if _is_geographic_clients_question(query):
        return (
            "השאלה היא על מדינות/אזורים שבהם ZOOZ עבדה. השתמש בעמודי לקוחות ופרויקטים ישירים. "
            "ציין רק מקומות שמופיעים במפורש במקורות, אל תטען שהלקוחות בישראל בלבד אם יש עדות לפעילות בחו\"ל, "
            "והבהר שהרשימה מבוססת על הדוגמאות המתועדות באתר ואינה בהכרח רשימה מלאה."
        )

    if _is_positioning_question(query):
        return (
            "השאלה היא על מיצוב. הסבר מהו המושג, למה הוא חשוב, ואיך הוא משפיע מעשית על בידול, מסר שיווקי, "
            "בחירת קהל או החלטות שיווקיות — רק ככל שהמקורות תומכים בכך. תן דוגמה קצרה אם יש במקור."
        )

    if _is_company_inference_question(query):
        return (
            "השאלה מבקשת מסקנה ברמת החברה. הסתמך קודם על דפי אודות, פרופיל החברה ושירותים ישירים. "
            "הפרד במפורש בין עובדה שמופיעה במקור לבין מסקנה סבירה, ואל תבסס מסקנה מרכזית על עלון ישן "
            "כשיש מקור חברה ישיר."
        )

    if intent == "team":
        return (
            "השאלה היא על אדם/צוות. תן פרופיל שימושי ב-2–3 פסקאות קצרות: תפקיד נוכחי, תחומי מומחיות, "
            "ורקע/פעילות מקצועית רלוונטית שמופיעים במקורות. אל תסתפק במשפט אחד, אבל שמור על תשובה ממוקדת."
        )

    if intent == "services_overview":
        return (
            "השאלה היא על שירותי החברה באופן כללי. הסתמך בראש ובראשונה על פרופיל החברה/אודות. "
            "הצג את תחומי-העל בקצרה, והוסף משפט אחד על הערך או המטרה של כל תחום. אל תעמיס פרטי משנה."
        )

    if intent == "contact":
        return "השאלה היא על יצירת קשר. תן רק את פרטי הקשר הישירים, בלי הרחבות מיותרות."

    if intent == "clients":
        return (
            "השאלה היא על לקוחות או ארגונים שעבדו עם ZOOZ. השתמש רק בעמודי לקוחות/פרויקטים ישירים, "
            "ואל תציג כחלק מהלקוחות חברות שמופיעות רק ברקע תעסוקתי של יועץ."
        )

    if intent == "triz":
        return (
            "השאלה היא על TRIZ. אל תטען ש-ZOOZ פיתחה את TRIZ או את I-TRIZ אלא אם מקור אומר זאת במפורש. "
            "הסבר את ההקשר לשירותי ZOOZ רק על בסיס המקורות."
        )

    if intent == "systematic_innovation":
        return (
            "השאלה היא על חדשנות שיטתית. הסבר את הרעיון, איך הוא מיושם בפועל ולמה משתמשים בו, "
            "ורק אז ציין שיטות או כלים שהמקורות אומרים במפורש ש-ZOOZ משתמשת בהם או מלמדת אותם."
        )

    return (
        "ענה ישירות ובצורה שימושית. ברירת המחדל היא 2–3 פסקאות קצרות, בערך 120–220 מילים בסך הכול: "
        "קודם תשובה ישירה, אחר כך הקשר/משמעות, ולבסוף יישום או דוגמה אם המקורות מאפשרים. "
        "אם נדרשת רשימה, השתמש לכל היותר ב-6 סעיפים קצרים. אל תאריך בכוח ואל תמציא."
    )


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
        document = _trim_context(item.get("document", ""))
        metadata = item.get("metadata") or {}
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
        if explicit_source and any(term in combined for term in explicit_price_terms) and currency_or_amount.search(combined):
            return True
    return False


def _augment_retrieval(collection, ranked, query, intent, mixed_pricing, company_inference):
    extra_queries = []

    if company_inference or mixed_pricing:
        extra_queries.append("אודות ZOOZ פרופיל החברה שירותים ייעוץ והדרכה")

    if intent == "team":
        extra_queries.append("ארי מנור מנכ״ל ZOOZ יועץ שיווקי אסטרטגיה חדשנות סדנאות ניסיון מקצועי")

    if _is_inventive_tools_question(query):
        extra_queries.append(
            "חשיבה המצאתית שישה כלי חשיבה הכפלה חלוקה החסרה איחוד הוספת מימד התאמה לסביבה"
        )

    if _is_multiplication_tool_question(query):
        extra_queries.append(
            "הכפלה כלי חשיבה המצאתית לבחור מרכיב להכפיל אותו למצוא תועלת מוצר שירות"
        )

    if _is_geographic_clients_question(query):
        extra_queries.append("לקוחות ZOOZ פרויקטים בחו״ל בעולם הונג קונג פעילות בינלאומית")

    if _is_positioning_question(query):
        extra_queries.append("מיצוב שיווקי בידול אסטרטגיה שיווקית ZOOZ מהו למה חשוב")

    merged = ranked
    for extra_query in extra_queries:
        extra_ranked = query_collection(
            collection,
            extra_query,
            candidate_k=RETRIEVAL_CANDIDATES,
            top_k=CONTEXT_CHUNKS,
        )
        merged = _merge_ranked_results(merged, extra_ranked, limit=14)

    return merged


def ask_zooz(query):
    start_time = time.time()

    try:
        intent = classify_query(query)
        mixed_pricing = _is_mixed_pricing_question(query)
        company_inference = _is_company_inference_question(query)

        if intent == "out_of_scope":
            answer = "אני יכול לעזור רק בנושאים הקשורים ל-ZOOZ."
            duration = round(time.time() - start_time, 2)
            log_to_csv(query, answer, duration)
            return answer, []

        if intent != "pricing":
            guarded_answer = _unsupported_fact_response(query)
            if guarded_answer:
                duration = round(time.time() - start_time, 2)
                log_to_csv(query, guarded_answer, duration)
                return guarded_answer, []

        if intent == "contact" and "כתובת" not in query:
            answer = (
                f"אפשר ליצור קשר עם ZOOZ בטלפון {CONTACT_PHONE} או במייל {CONTACT_EMAIL}. "
                "פרטים נוספים מופיעים בדף יצירת הקשר של החברה."
            )
            duration = round(time.time() - start_time, 2)
            log_to_csv(query, answer, duration)
            return answer, [CONTACT_URL]

        collection = load_collection()
        retrieval_query = _strip_pricing_terms(query) if mixed_pricing else query
        ranked = query_collection(
            collection,
            retrieval_query,
            candidate_k=RETRIEVAL_CANDIDATES,
            top_k=CONTEXT_CHUNKS,
        )
        ranked = _augment_retrieval(
            collection,
            ranked,
            query,
            intent,
            mixed_pricing,
            company_inference,
        )

        if is_pricing_query(query) and not mixed_pricing and not has_current_pricing_evidence(ranked):
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

ענה רק על בסיס המקורות שסופקו מאתר ZOOZ. המטרה העליונה היא דיוק, אך התשובה צריכה גם להיות שימושית למשתמש.

כללים:
1. ענה בעברית ברורה וטבעית. ברירת המחדל היא 2–3 פסקאות קצרות ומשמעותיות, בדרך כלל 120–220 מילים בסך הכול.
2. בשאלה על מושג, כלי, שירות או אדם: תן קודם תשובה ישירה, אחר כך הקשר/משמעות, ולבסוף יישום, דוגמה או רקע שימושי אם המקורות תומכים בכך.
3. אל תשתמש בידע חיצוני ואל תשלים פרטים חסרים מהשערה.
4. עובדה ספציפית כמו מחיר, שם, תפקיד, לקוח, תאריך, מספר או שירות מותרת רק אם היא מופיעה במפורש במקורות.
5. אם העובדה המדויקת לא מופיעה, אמור שאין לך מידע מדויק עליה במקורות של ZOOZ.
6. אל תאשר הנחה לא נתמכת ואל תמציא שירותים, לקוחות, מחירים או נתונים.
7. העדף דפי אודות, שירותים, צוות, קשר, לקוחות ועמודי תחום ישירים על פני אזכורים מקריים במאמרים או עלונים ישנים.
8. אל תענה על נושאים שאינם קשורים ל-ZOOZ ואל תזכיר ציוני retrieval או פרטים פנימיים של המערכת.
9. מחיר שמופיע במאמר/עלון/דוגמה ישנה אינו מחירון שירותי ZOOZ.
10. אם השאלה כוללת כמה חלקים, ענה לכל חלק. אם חסר מידע לגבי חלק אחד, ציין זאת רק לגבי אותו חלק והמשך לענות על שאר החלקים.
11. אל תאריך בכוח. הימנע מחזרות, ואל תוסיף סעיפים שאינם נחוצים רק כדי להאריך את התשובה.
12. כשאפשר, תן למשתמש \"מה עושים עם המידע\" — צעד, שימוש, דוגמה או תועלת — אך רק אם הדבר מבוסס במקורות.
13. אם נדרשת רשימה, השתמש לכל היותר ב-6 סעיפים קצרים. ודא שהתשובה מסתיימת במשפט שלם ולא באמצע סעיף או משפט.

מיקוד לשאלה:
{guidance}

מקורות:
{context}

שאלה:
{query}

תשובה:"""

        client = get_groq_client()
        response = _create_completion(client, prompt)
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Groq returned empty answer content")

        duration = round(time.time() - start_time, 2)
        log_to_csv(query, answer, duration)
        return answer, sources

    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"CHATBOT ERROR: {repr(exc)}")
        if _is_rate_limit_error(exc):
            error_msg = "השירות עמוס זמנית בגלל מגבלת שימוש. אפשר לנסות שוב בעוד כמה דקות."
        else:
            error_msg = "אירעה שגיאה זמנית בשירות. אנא נסה שוב בעוד רגע."
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
