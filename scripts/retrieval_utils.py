import hashlib
import re
from collections import defaultdict

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


def get_embedding_function():
    """Return the exact multilingual embedding function used for both indexing and querying."""
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


def query_terms(query):
    return [token for token in tokens(query) if token not in STOPWORDS and len(token) > 1]


def _exact_phrase(query):
    terms = query_terms(query)
    return " ".join(terms) if len(terms) >= 2 else ""


def query_collection(collection, query, candidate_k=40, top_k=8):
    """Retrieve a broad vector candidate set and rerank it for factual ZOOZ QA.

    Vector similarity remains the main signal, but Hebrew/English language alignment,
    exact phrase matches and lexical/title overlap help prevent generic navigation
    chunks from outranking the page that actually contains the requested fact.
    """
    count = collection.count()
    if count == 0:
        return []

    candidate_k = max(top_k, min(candidate_k, count))
    results = collection.query(
        query_texts=[query],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    q_terms = query_terms(query)
    q_set = set(q_terms)
    phrase = _exact_phrase(query).lower()
    query_lang = detect_language(query)

    candidates = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        title = normalize(metadata.get("title", ""))
        document = normalize(document)
        combined = f"{title} {document}".lower()
        combined_tokens = set(tokens(combined))
        title_tokens = set(tokens(title))

        lexical_overlap = (
            len(q_set & combined_tokens) / len(q_set) if q_set else 0.0
        )
        title_overlap = (
            len(q_set & title_tokens) / len(q_set) if q_set else 0.0
        )
        phrase_bonus = 0.35 if phrase and phrase in combined else 0.0

        chunk_lang = metadata.get("language") or detect_language(document)
        language_adjustment = 0.0
        if query_lang == "he":
            if chunk_lang == "he":
                language_adjustment = 0.10
            elif chunk_lang == "en":
                language_adjustment = -0.12

        # Cosine distance is lower-is-better; convert it into a higher-is-better base.
        vector_score = 1.0 - float(distance)
        rerank_score = (
            vector_score
            + 0.24 * lexical_overlap
            + 0.28 * title_overlap
            + phrase_bonus
            + language_adjustment
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
            "score": rerank_score,
            "lexical_overlap": lexical_overlap,
            "title_overlap": title_overlap,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Exact-content deduplication and a small per-page cap prevent one legacy page
    # or duplicated menu block from filling the whole LLM context.
    selected = []
    seen_content = set()
    per_url = defaultdict(int)

    for item in candidates:
        document_hash = hashlib.sha1(
            item["document"].encode("utf-8")
        ).hexdigest()
        if document_hash in seen_content:
            continue

        url = item["metadata"].get("url", "")
        if url and per_url[url] >= 3:
            continue

        seen_content.add(document_hash)
        if url:
            per_url[url] += 1
        selected.append(item)

        if len(selected) >= top_k:
            break

    return selected
