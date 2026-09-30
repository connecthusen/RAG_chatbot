import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.vectorstore.chroma_client import get_client, get_or_create_collection, get_collection_count
from app.graph.build_graph import build_rag_graph


SAMPLE_QUESTIONS = [
    "What is agentic AI?",
    "What are the key components of an AI agent?",
    "What is the capital of France?",  # should be refused — not in the document
]


def main():
    settings = get_settings()

    client = get_client(settings.chroma_persist_dir)
    collection = get_or_create_collection(client, settings.chroma_collection_name)

    count = get_collection_count(collection)
    print(f"Chroma collection count: {count}")
    if count == 0:
        print("Collection is empty — did ingestion actually run? Check scripts/ingest.py output.")
        sys.exit(1)

    graph = build_rag_graph(collection, settings)

    for question in SAMPLE_QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print('=' * 60)

        result = graph.invoke({"question": question})

        print(f"Confidence : {result['confidence_score']:.4f}")
        print(f"Chunks used: {len(result['retrieved_chunks'])}")
        print(f"Answer     : {result['answer']}")


if __name__ == "__main__":
    main()