import argparse

import chromadb

try:
    from scripts.retrieval_utils import (
        EMBEDDING_MODEL,
        get_embedding_function,
        query_collection,
    )
except ImportError:
    from retrieval_utils import (
        EMBEDDING_MODEL,
        get_embedding_function,
        query_collection,
    )

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zooz_knowledge"

DEFAULT_QUESTIONS = [
    "מה השירותים של ZOOZ?",
    "מה זה חדשנות שיטתית?",
    "מי זה ארי מנור?",
    "איך ליצור קשר עם ZOOZ?",
    "כמה עולים השירותים?",
    "מה ZOOZ יכולה לעשות עבורי?",
    "מי הלקוחות של ZOOZ?",
    "אילו סדנאות ZOOZ מציעה?",
    "במה ZOOZ עוסקת בתחום השיווק?",
    "מה ZOOZ מציעה בתחום פיתוח מנהלים?",
]


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def evaluate_question(collection, question, top_k, candidate_k):
    ranked = query_collection(
        collection,
        question,
        candidate_k=candidate_k,
        top_k=top_k,
    )

    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    for rank, item in enumerate(ranked, start=1):
        metadata = item["metadata"]
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        chunk_index = metadata.get("chunk_index", "")
        language = metadata.get("language", "")
        preview = " ".join(item["document"].split())[:450]

        print(
            f"\n#{rank} | rerank={item['score']:.4f} "
            f"| distance={item['distance']:.4f} "
            f"| lexical={item['lexical_overlap']:.2f} "
            f"| lang={language} | chunk={chunk_index}"
        )
        print(f"TITLE: {title}")
        print(f"URL:   {url}")
        print(f"TEXT:  {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect multilingual + reranked retrieval quality for ZOOZ."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional custom question. If omitted, the ZOOZ smoke-test set is used.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=40)
    args = parser.parse_args()

    questions = [" ".join(args.question)] if args.question else DEFAULT_QUESTIONS
    collection = load_collection()

    print(f"Collection contains {collection.count()} chunks")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Candidate pool: {args.candidate_k} -> showing top {args.top_k}")

    for question in questions:
        evaluate_question(
            collection,
            question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )


if __name__ == "__main__":
    main()
