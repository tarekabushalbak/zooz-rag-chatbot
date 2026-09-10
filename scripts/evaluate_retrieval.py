import argparse
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"

DEFAULT_QUESTIONS = [
    "מה השירותים של ZOOZ?",
    "מה זה חדשנות שיטתית?",
    "מי זה ארי מנור?",
    "איך ליצור קשר עם ZOOZ?",
    "כמה עולים השירותים?",
    "מה ZOOZ יכולה לעשות עבורי?",
]


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def evaluate_question(collection, question, top_k):
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("\n" + "=" * 90)
    print(f"QUESTION: {question}")
    print("=" * 90)

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        chunk_index = metadata.get("chunk_index", "")
        preview = " ".join(document.split())[:350]

        print(f"\n#{rank} | distance={distance:.4f} | chunk={chunk_index}")
        print(f"TITLE: {title}")
        print(f"URL:   {url}")
        print(f"TEXT:  {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB retrieval quality for the ZOOZ RAG index."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional custom question. If omitted, a small ZOOZ smoke-test set is used.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = [" ".join(args.question)] if args.question else DEFAULT_QUESTIONS
    collection = load_collection()

    print(f"Collection contains {collection.count()} chunks")
    for question in questions:
        evaluate_question(collection, question, args.top_k)


if __name__ == "__main__":
    main()
