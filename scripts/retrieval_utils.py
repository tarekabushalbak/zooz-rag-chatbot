import hashlib
import re
from collections import defaultdict
from functools import lru_cache
from urllib.parse import urlparse

from chromadb.utils import embedding_functions

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
TOKEN_RE = re.compile(r"[\u0590-\u05FFA-Za-z0-9]+")

STOPWORDS = {
    "מה", "מי", "זה", "זו", "זאת", "של", "את", "על", "עם", "איך", "כמה",
    "האם", "יש", "אין", "למה", "מתי", "איפה", "איזה", "איזו", "אילו",
    "חברה", "החברה", "חברת", "zooz", "זוז",
    "the", "a", "an", "of", "to", "for", "is", "are", "what", "who", "how",
}

DOMAIN_TERMS = {
    "שיווק", "חדשנות", "אסטרטגיה", "מנהלים", "עובדים", "סדנאות", "סדנה",
    "הדרכה", "לקוחות", "לקוח", "קשר", "טלפון", "מייל", "מחיר", "עלות",
}


@lru_cache(maxsize=1)
def get_embedding_function():
    """Create the multilingual embedding model once per process.

    Loading SentenceTransformer for every question was the main local latency source.
    Reusing one instance keeps the embedding model warm across regression tests and
    chatbot requests.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def normalize(text):
    return " ".join((text or "").split()).strip()


def detect_language(text):
    text = text or ""
    hebrew = len(HEBREW_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if hebrew > latin * 1.2:
        return "he"
    if latin > hebrew * 1.2:
        return "en"
    return "mixed"


def tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _hebrew_variants(token):
    """Return conservative lexical variants for Hebrew prefixes/plural endings."""
    variants = {token}
    if not HEBREW_RE.search(token):
        return variants

    bases = {token}
    if len(token) >= 5 and token[0] in "הובכלש":
        bases.add(token[1:])

    for base in list(bases):
        variants.add(base)
        if len(base) >= 5 and base.endswith("ים"):
            variants.add(base[:-2])
        if len(base) >= 5 and base.endswith("ות"):
            variants.add(base[:-2])
        if len(base) >= 5 and base.endswith("י"):
            variants.add(base[:-1])

    return {item for item in variants if len(item) > 1}


def lexical_token_set(text):
    result = set()
    for token in tokens(text):
        result.update(_hebrew_variants(token))
    return result


def query_terms(query):
    return [token for token in tokens(query) if token not in STOPWORDS and len(token) > 1]


def query_term_set(query):
    result = set()
    for token in query_terms(query):
        result.update(_hebrew_variants(token))
    return result


def _exact_phrase(query):
    terms = query_terms(query)
    return " ".join(terms) if len(terms) >= 2 else ""


def classify_query(query):
    q = normalize(query).lower()
    q_tokens = set(tokens(q))

    # Obvious non-ZOOZ requests should never be sent into retrieval where an old
    # newsletter can accidentally contain a matching word and tempt the LLM to answer.
    if any(term in q for term in [
        "מזג האוויר", "תחזית מזג", "מי ניצח", "משחק אתמול", "תספר לי בדיחה",
        "ספר לי בדיחה", "פוליטיקה", "מה דעתך על פוליט",
    ]):
        return "out_of_scope"

    if any(term in q for term in ["כמה עולה", "כמה עולים", "מה המחיר", "מחיר", "עלות", "תמחור"]):
        return "pricing"

    if any(term in q for term in [
        "יצירת קשר", "ליצור קשר", "טלפון", "אימייל", "אי-מייל", "מייל", "כתובת",
        "איך מדברים איתכם", "איך מדברים אתכם", "לדבר איתכם", "לדבר אתכם",
        "לפנות אליכם", "איך פונים", "איפה פונים", "איפה אפשר לפנות", "אפשר לפנות",
    ]):
        return "contact"

    if "ארי מנור" in q or "צוות" in q or "מנכ\"ל" in q or "מנכל" in q or q.strip() == "מי ארי?":
        return "team"

    if (
        "לקוח" in q
        or "לקוחות" in q
        or any(term in q for term in [
            "עבדה עם", "עובדת עם", "עבדו עם", "עובדים עם",
            "ארגונים גדולים", "ארגון גדול", "חברות גדולות", "חברה גדולה",
        ])
    ):
        return "clients"

    if any(term in q for term in ["סדנה", "סדנא", "סדנאות", "הרצאה", "הרצאות"]):
        return "workshops"

    if "פיתוח מנהלים" in q:
        return "management_development"

    if "triz" in q:
        return "triz"

    if any(term in q for term in ["חדשנות שיטתית", "שיטות חדשנות", "שיטות לחדשנות", "שיטות של חדשנות"]):
        return "systematic_innovation"

    if "שיווק" in q:
        return "marketing"

    service_words = any(term in q for term in [
        "שירות", "שרות", "מציעה", "מציע", "מספקת", "מספק", "יכולה לעשות",
        "יכול לעשות", "מה עושה", "במה עוסקת", "במה עוסק",
    ])
    zooz_named = "zooz" in q or "זוז" in q
    if zooz_named and any(term in q for term in ["עושה", "עושים", "עוסקת", "עוסק"]):
        service_words = True

    if service_words and not (q_tokens & DOMAIN_TERMS):
        return "services_overview"
    if service_words:
        return "services_overview"

    return "general"


def expand_query(query, intent=None):
    intent = intent or classify_query(query)
    additions = {
        "services_overview": "אודות ZOOZ פרופיל החברה שירותים ייעוץ והדרכה",
        "team": "הצוות של ZOOZ אודות מנכ\"ל",
        "contact": "יצירת קשר ZOOZ",
        "clients": "לקוחות ZOOZ פרויקטים",
        "workshops": "סדנאות הדרכות ZOOZ",
        "management_development": "פיתוח מנהלים הדרכה ZOOZ",
        "systematic_innovation": "חדשנות שיטתית שיטות ניהול חדשנות ZOOZ",
        "triz": "TRIZ I-TRIZ Ideation חדשנות ZOOZ",
        "marketing": "שיווק ייעוץ שיווקי אסטרטגיה ZOOZ",
    }
    suffix = additions.get(intent, "")
    return f"{query} {suffix}".strip() if suffix else query


def _url_path(url):
    return urlparse(url or "").path.lower()


def _intent_url_adjustment(intent, url, title, query):
    path = _url_path(url)
    title_l = (title or "").lower()
    q = normalize(query).lower()
    score = 0.0

    if "/lazooz/" in path and "עלון" not in q:
        score -= 0.08
    if "_article" in path and "מאמר" not in q:
        score -= 0.04
    if "/news" in path and intent in {"services_overview", "team", "contact", "pricing"}:
        score -= 0.10

    if intent == "services_overview":
        if path in {"/about.shtml", "/about_profile.shtml"}:
            score += 0.58
        elif path == "/personel.shtml":
            score += 0.35
        elif "marketing_services" in path or "services" in path:
            score += 0.28
        if "פרופיל החברה" in title_l or "אודות" in title_l:
            score += 0.18

    elif intent == "team":
        if path == "/about_team.shtml":
            score += 0.70
        elif path == "/marketing_article13.shtml":
            score += 0.35
        if "ארי מנור" in title_l or "הצוות" in title_l:
            score += 0.28

    elif intent == "contact":
        if path == "/contact.shtml":
            score += 0.75
        elif path in {"/about.shtml", "/about_profile.shtml"}:
            score += 0.22

    elif intent == "clients":
        if "news_clients" in path or "about_clients" in path or "clients" in path:
            score += 0.65
        if "personel" in path or "_article" in path:
            score -= 0.18

    elif intent == "workshops":
        if "workshop" in path or "training" in path:
            score += 0.40

    elif intent == "management_development":
        if "personel" in path or "management" in path:
            score += 0.32

    elif intent == "systematic_innovation":
        if "innovation-methods" in path:
            score += 0.48
        elif "innovation" in path:
            score += 0.18

    elif intent == "triz":
        if path == "/marketing_article14.shtml":
            score += 0.72
        elif "innovation-methods" in path:
            score += 0.58
        elif "triz" in path or "innovation" in path:
            score += 0.20

    elif intent == "marketing":
        if "marketing_services" in path or "1-marketing" in path:
            score += 0.34
        elif "marketing" in path:
            score += 0.12

    elif intent == "pricing":
        if "/lazooz/" in path or "_article" in path or "/news" in path:
            score -= 0.35
        if "price" in path or "pricing" in path or "מחיר" in title_l or "תמחור" in title_l:
            score += 0.50

    return score


def is_pricing_query(query):
    return classify_query(query) == "pricing"


def query_collection(collection, query, candidate_k=80, top_k=8):
    """Retrieve a broad multilingual candidate set and rerank it for ZOOZ QA."""
    count = collection.count()
    if count == 0:
        return []

    intent = classify_query(query)
    retrieval_query = expand_query(query, intent)
    candidate_k = max(top_k, min(candidate_k, count))
    results = collection.query(
        query_texts=[retrieval_query],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    q_set = query_term_set(query)
    phrase = _exact_phrase(query).lower()
    query_lang = detect_language(query)

    candidates = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        title = normalize(metadata.get("title", ""))
        url = metadata.get("url", "")
        document = normalize(document)
        combined = f"{title} {document}".lower()
        combined_tokens = lexical_token_set(combined)
        title_tokens = lexical_token_set(title)

        lexical_overlap = len(q_set & combined_tokens) / len(q_set) if q_set else 0.0
        title_overlap = len(q_set & title_tokens) / len(q_set) if q_set else 0.0
        phrase_bonus = 0.42 if phrase and phrase in combined else 0.0

        chunk_lang = metadata.get("language") or detect_language(document)
        language_adjustment = 0.0
        if query_lang == "he":
            if chunk_lang == "he":
                language_adjustment = 0.10
            elif chunk_lang == "en":
                language_adjustment = -0.12

        vector_score = 1.0 - float(distance)
        url_adjustment = _intent_url_adjustment(intent, url, title, query)
        rerank_score = (
            vector_score
            + 0.30 * lexical_overlap
            + 0.34 * title_overlap
            + phrase_bonus
            + language_adjustment
            + url_adjustment
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
            "score": rerank_score,
            "lexical_overlap": lexical_overlap,
            "title_overlap": title_overlap,
            "intent": intent,
            "retrieval_query": retrieval_query,
            "url_adjustment": url_adjustment,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    seen_content = set()
    per_url = defaultdict(int)

    for item in candidates:
        document_hash = hashlib.sha1(item["document"].encode("utf-8")).hexdigest()
        if document_hash in seen_content:
            continue

        url = item["metadata"].get("url", "")
        if url and per_url[url] >= 2:
            continue

        seen_content.add(document_hash)
        if url:
            per_url[url] += 1
        selected.append(item)

        if len(selected) >= top_k:
            break

    return selected