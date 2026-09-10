import hashlib
import json
import os
import re

import chromadb

INPUT_FILE = "data/pages.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"

TARGET_CHUNK_CHARS = 1000
MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 180
OVERLAP_SENTENCES = 2
BATCH_SIZE = 100


def normalize_space(text):
    return " ".join((text or "").split())


def split_sentences(text):
    """Sentence-aware fallback for old pages.json files that contain flat text."""
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [normalize_space(part) for part in parts if normalize_space(part)]
    return sentences or [text]


def split_oversized_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split a long paragraph without cutting blindly in the middle of sentences."""
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

            # Last-resort split for a very long sentence/HTML artifact.
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


def chunk_flat_text(text):
    """Create coherent chunks from legacy flattened content using sentence boundaries."""
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
    """Chunk crawler blocks while keeping headings with the content that follows them."""
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

    # Merge tiny trailing chunks when possible.
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK_CHARS:
            combined = f"{merged[-1]}\n{chunk}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                merged[-1] = combined
                continue
        merged.append(chunk)

    return merged


def create_chunks(page):
    blocks = page.get("blocks") or []
    if blocks:
        chunks = chunk_structured_blocks(blocks)
    else:
        chunks = chunk_flat_text(page.get("text", ""))

    title = normalize_space(page.get("title", ""))
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Repeating the page title in the indexed text gives retrieval a useful
        # semantic anchor, especially when a chunk starts in the middle of an article.
        if title and not chunk.startswith(title):
            indexed_text = f"{title}\n{chunk}"
        else:
            indexed_text = chunk

        result.append(indexed_text)

    return result


def recreate_collection(client):
    """Build from a clean collection so stale chunks from previous runs cannot survive."""
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    return client.create_collection(name=COLLECTION_NAME)


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

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = recreate_collection(client)

    total_chunks = 0
    batch_documents = []
    batch_metadatas = []
    batch_ids = []

    for page_number, page in enumerate(pages, start=1):
        url = page.get("url", "")
        title = normalize_space(page.get("title", ""))
        text = page.get("text", "")

        if not url or not text or len(text) < 50:
            continue

        chunks = create_chunks(page)
        page_id = stable_page_id(url)

        for chunk_index, chunk in enumerate(chunks):
            batch_documents.append(chunk)
            batch_metadatas.append({
                "url": url,
                "title": title,
                "chunk_index": chunk_index,
                "chunk_chars": len(chunk),
            })
            batch_ids.append(f"{page_id}__chunk_{chunk_index}")
            total_chunks += 1

            if len(batch_documents) >= BATCH_SIZE:
                add_batch(collection, batch_documents, batch_metadatas, batch_ids)
                batch_documents, batch_metadatas, batch_ids = [], [], []

        print(f"[{page_number}/{len(pages)}] {title[:60]} — {len(chunks)} chunks")

    add_batch(collection, batch_documents, batch_metadatas, batch_ids)

    print(f"\nDone! Created {total_chunks} structure-aware chunks")
    print(f"ChromaDB saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
