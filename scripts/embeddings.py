import hashlib
import json
import os
import re
from collections import Counter

import chromadb

try:
    from scripts.retrieval_utils import (
        EMBEDDING_MODEL,
        detect_language,
        get_embedding_function,
        normalize,
    )
except ImportError:
    from retrieval_utils import (
        EMBEDDING_MODEL,
        detect_language,
        get_embedding_function,
        normalize,
    )

INPUT_FILE = "data/pages.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"

TARGET_CHUNK_CHARS = 1000
MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 180
OVERLAP_SENTENCES = 2
BATCH_SIZE = 100

# A block repeated on many different pages is almost always legacy navigation,
# footer/contact boilerplate or another site-wide template fragment.
BOILERPLATE_MIN_PAGES = 15
BOILERPLATE_PAGE_RATIO = 0.02


def normalize_space(text):
    return normalize(text)


def split_sentences(text):
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [normalize_space(part) for part in parts if normalize_space(part)]
    return sentences or [text]


def split_oversized_text(text, max_chars=MAX_CHUNK_CHARS):
    sentences = split_sentences(text)
    if not sentences:
        return []

    result = []
    current = []
    current_len = 0

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                result.append(" ".join(current))
                current = []
                current_len = 0

            start = 0
            while start < len(sentence):
                piece = sentence[start:start + max_chars].strip()
                if piece:
                    result.append(piece)
                start += max_chars
            continue

        projected = current_len + len(sentence) + (1 if current else 0)
        if current and projected > max_chars:
            result.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = projected

    if current:
        result.append(" ".join(current))

    return result


def build_boilerplate_set(pages):
    """Detect non-heading blocks repeated across many pages.

    ZOOZ uses older table-based templates, so navigation often appears as ordinary
    text cells rather than <nav>/<header>. Cross-page frequency is much more reliable
    than a CSS-selector-only cleanup for this site.
    """
    frequencies = Counter()

    for page in pages:
        seen_on_page = set()
        for block in page.get("blocks") or []:
            if block.get("type") == "heading":
                continue
            text = normalize_space(block.get("text", ""))
            if len(text) < 20:
                continue
            seen_on_page.add(text)
        frequencies.update(seen_on_page)

    threshold = max(
        BOILERPLATE_MIN_PAGES,
        int(len(pages) * BOILERPLATE_PAGE_RATIO),
    )
    boilerplate = {
        text for text, page_count in frequencies.items()
        if page_count >= threshold
    }
    print(
        f"Detected {len(boilerplate)} repeated boilerplate blocks "
        f"(present on at least {threshold} pages)"
    )
    return boilerplate


def filtered_blocks(page, boilerplate):
    result = []
    previous = None

    for block in page.get("blocks") or []:
        text = normalize_space(block.get("text", ""))
        if not text:
            continue

        if block.get("type") != "heading" and text in boilerplate:
            continue

        # Drop repeated adjacent blocks and tiny fragments left by legacy tables.
        if text == previous:
            continue
        previous = text

        result.append({
            "type": block.get("type", "text"),
            "text": text,
        })

    return result


def chunk_flat_text(text):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        pieces = split_oversized_text(sentence)
        for piece in pieces:
            projected = current_len + len(piece) + (1 if current else 0)
            if current and projected > TARGET_CHUNK_CHARS:
                chunk = " ".join(current).strip()
                if chunk:
                    chunks.append(chunk)

                overlap = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
                current = overlap[:]
                current_len = len(" ".join(current))

            current.append(piece)
            current_len = len(" ".join(current))

            if current_len >= MAX_CHUNK_CHARS:
                chunks.append(" ".join(current).strip())
                current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
                current_len = len(" ".join(current))

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            if chunks and len(chunk) < MIN_CHUNK_CHARS:
                merged = f"{chunks[-1]} {chunk}".strip()
                if len(merged) <= MAX_CHUNK_CHARS:
                    chunks[-1] = merged
                else:
                    chunks.append(chunk)
            else:
                chunks.append(chunk)

    return chunks


def chunk_structured_blocks(blocks):
    chunks = []
    current_heading = ""
    current_parts = []

    def flush():
        nonlocal current_parts
        if not current_parts:
            return
        chunk = "\n".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []

    for block in blocks or []:
        text = normalize_space(block.get("text", ""))
        if not text:
            continue

        if block.get("type") == "heading":
            if current_parts and len("\n".join(current_parts)) >= MIN_CHUNK_CHARS:
                flush()
            current_heading = text
            continue

        pieces = split_oversized_text(text)
        for piece in pieces:
            prefix = [current_heading] if current_heading else []
            candidate_parts = current_parts + [piece]
            candidate = "\n".join(candidate_parts)

            if current_parts and len(candidate) > TARGET_CHUNK_CHARS:
                flush()
                current_parts = prefix + [piece]
            else:
                if not current_parts and prefix:
                    current_parts.extend(prefix)
                current_parts.append(piece)

            if len("\n".join(current_parts)) >= MAX_CHUNK_CHARS:
                flush()

    flush()

    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK_CHARS:
            combined = f"{merged[-1]}\n{chunk}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                merged[-1] = combined
                continue
        merged.append(chunk)

    return merged


def create_chunks(page, boilerplate):
    blocks = filtered_blocks(page, boilerplate)
    if blocks:
        chunks = chunk_structured_blocks(blocks)
    else:
        # Legacy fallback only. Structured crawler output should normally use blocks.
        chunks = chunk_flat_text(page.get("text", ""))

    title = normalize_space(page.get("title", ""))
    result = []
    seen = set()

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if title and not chunk.startswith(title):
            indexed_text = f"{title}\n{chunk}"
        else:
            indexed_text = chunk

        indexed_text = indexed_text.strip()
        content_hash = hashlib.sha1(indexed_text.encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        result.append(indexed_text)

    return result


def recreate_collection(client):
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def stable_page_id(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def add_batch(collection, batch_documents, batch_metadatas, batch_ids):
    if not batch_documents:
        return
    collection.add(
        documents=batch_documents,
        metadatas=batch_metadatas,
        ids=batch_ids,
    )


def main():
    print("Loading ZOOZ pages...")
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        pages = json.load(file)
    print(f"Loaded {len(pages)} pages")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    boilerplate = build_boilerplate_set(pages)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = recreate_collection(client)

    total_chunks = 0
    duplicate_chunks_skipped = 0
    global_chunk_hashes = set()
    batch_documents = []
    batch_metadatas = []
    batch_ids = []

    for page_number, page in enumerate(pages, start=1):
        url = page.get("url", "")
        title = normalize_space(page.get("title", ""))
        text = page.get("text", "")

        if not url or not text or len(text) < 50:
            continue

        chunks = create_chunks(page, boilerplate)
        page_id = stable_page_id(url)
        kept_for_page = 0

        for chunk_index, chunk in enumerate(chunks):
            content_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            if content_hash in global_chunk_hashes:
                duplicate_chunks_skipped += 1
                continue
            global_chunk_hashes.add(content_hash)

            batch_documents.append(chunk)
            batch_metadatas.append({
                "url": url,
                "title": title,
                "chunk_index": chunk_index,
                "chunk_chars": len(chunk),
                "language": detect_language(chunk),
                "is_english_path": int("/eng/" in url.lower()),
            })
            batch_ids.append(f"{page_id}__chunk_{chunk_index}")
            total_chunks += 1
            kept_for_page += 1

            if len(batch_documents) >= BATCH_SIZE:
                add_batch(collection, batch_documents, batch_metadatas, batch_ids)
                batch_documents, batch_metadatas, batch_ids = [], [], []

        print(
            f"[{page_number}/{len(pages)}] {title[:60]} — "
            f"{kept_for_page} indexed chunks"
        )

    add_batch(collection, batch_documents, batch_metadatas, batch_ids)

    print(f"\nDone! Created {total_chunks} unique structure-aware chunks")
    print(f"Skipped {duplicate_chunks_skipped} exact duplicate chunks")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"ChromaDB saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
