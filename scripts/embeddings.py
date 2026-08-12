import json
import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "data/pages.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def main():
    print("טוען את הדפים...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)
    print(f"נטענו {len(pages)} דפים")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    total_chunks = 0
    for page in pages:
        url = page["url"]
        title = page["title"]
        text = page["text"]
        
        if not text or len(text) < 50:
            continue
        
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{url}__chunk_{i}"
            try:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"url": url, "title": title, "chunk_index": i}],
                    ids=[chunk_id]
                )
                total_chunks += 1
            except Exception as e:
                pass
        
        print(f"עובד: {title[:50]} — {len(chunks)} chunks")

    print(f"\nסיום! נוצרו {total_chunks} chunks")
    print(f"ChromaDB נשמר ב-{CHROMA_DIR}")

if __name__ == "__main__":
    main()