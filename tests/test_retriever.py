import shutil

from app.config import get_settings
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_client import (
    get_client,
    get_or_create_collection,
    add_children_to_store,
    reset_collection,
)
from app.vectorstore.parent_store import save_parents, reset_parent_store
from app.retriever.retriever import retrieve

TEST_PERSIST_DIR = "data/vector_db_TEST"
TEST_COLLECTION_NAME = "test_collection_retriever"
TEST_PARENT_STORE_PATH = "data/vector_db_TEST/parents_test_retriever.json"


def test_retriever():
    settings = get_settings()

    client = get_client(TEST_PERSIST_DIR)
    reset_collection(client, TEST_COLLECTION_NAME)
    reset_parent_store(TEST_PARENT_STORE_PATH)
    collection = get_or_create_collection(client, TEST_COLLECTION_NAME)

    # 2 parents, 4 children (parent_0 has 2 children, to test de-duplication)
    child_ids = ["child_0", "child_1", "child_2", "child_3"]
    child_texts = [
        "Agentic AI systems can autonomously plan multi-step tasks.",
        "They use tools and external APIs to complete objectives.",
        "RAG combines retrieval with generation for grounded answers.",
        "Paris is the capital city of France.",
    ]
    parent_ids = ["parent_0", "parent_0", "parent_1", "parent_2"]

    unique_parent_ids = ["parent_0", "parent_1", "parent_2"]
    unique_parent_texts = [
        "Agentic AI systems can autonomously plan multi-step tasks. They use tools and external APIs to complete objectives.",
        "RAG combines retrieval with generation for grounded answers. It reduces hallucination by grounding responses in source documents.",
        "Paris is the capital city of France, known for the Eiffel Tower.",
    ]

    print("\nSetting up test data ")
    save_parents(TEST_PARENT_STORE_PATH, unique_parent_ids, unique_parent_texts)
    embeddings = embed_texts(child_texts, settings.embedding_model)
    add_children_to_store(collection, child_ids, child_texts, embeddings, parent_ids)

    print("\n Query: 'What is agentic AI?'")
    results = retrieve(
        question="What is agentic AI?",
        collection=collection,
        embedding_model=settings.embedding_model,
        parent_store_path=TEST_PARENT_STORE_PATH,
        rerank_model=settings.rerank_model,
        retrieval_candidates=4,  # small for this test — only 4 children exist
        top_k=4,
        hf_token=settings.hf_token,
    )

    print(f"Number of unique parent contexts returned: {len(results)}")
    for i, ctx in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  parent_id        : {ctx.parent_id}")
        print(f"  similarity_score : {ctx.similarity_score:.4f}")
        print(f"  matched_child    : {ctx.matched_child_text}")
        print(f"  parent_text      : {ctx.parent_text[:100]}...")

    print("\n Checks ")
    print(f"Top result is about agentic AI (parent_0): {results[0].parent_id == 'parent_0'}")
    print(f"No duplicate parent_ids in results: {len(results) == len(set(r.parent_id for r in results))}")
    print(f"Unrelated 'Paris' result ranks lowest: {results[-1].parent_id == 'parent_2'}")

    # Cleanup
    reset_collection(client, TEST_COLLECTION_NAME)
    reset_parent_store(TEST_PARENT_STORE_PATH)
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)


if __name__ == "__main__":
    test_retriever()