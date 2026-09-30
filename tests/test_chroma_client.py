from app.embeddings.embedder import embed_texts
from app.config import get_settings
from app.vectorstore.chroma_client import (
    get_client,
    get_or_create_collection,
    add_children_to_store,
    get_collection_count,
    reset_collection,
)
from app.vectorstore.parent_store import save_parents, get_parent_text, reset_parent_store

TEST_PERSIST_DIR = "data/vector_db_TEST"
TEST_COLLECTION_NAME = "test_collection"
TEST_PARENT_STORE_PATH = "data/vector_db_TEST/parents_test.json"


def test_chroma_client():
    settings = get_settings()  # only used for embedding_model, not storage paths

    print(f"\n[TEST-ONLY] Chroma persist dir : {TEST_PERSIST_DIR}")
    print(f"[TEST-ONLY] Collection name    : {TEST_COLLECTION_NAME}")
    print(f"[TEST-ONLY] Parent store path  : {TEST_PARENT_STORE_PATH}")

    client = get_client(TEST_PERSIST_DIR)
    reset_collection(client, TEST_COLLECTION_NAME)
    reset_parent_store(TEST_PARENT_STORE_PATH)
    collection = get_or_create_collection(client, TEST_COLLECTION_NAME)

    print(f"Initial count: {get_collection_count(collection)}")

    child_ids = ["child_0", "child_1", "child_2"]
    child_texts = [
        "Agentic AI systems can autonomously plan multi-step tasks.",
        "They use tools and external APIs to complete objectives.",
        "RAG combines retrieval with generation for grounded answers.",
    ]
    parent_ids = ["parent_0", "parent_0", "parent_1"]

    unique_parent_ids = ["parent_0", "parent_1"]
    unique_parent_texts = [
        "Agentic AI systems can autonomously plan multi-step tasks. They use tools and external APIs to complete objectives.",
        "RAG combines retrieval with generation for grounded answers. It reduces hallucination by grounding responses in source documents.",
    ]

    print("\nSaving parent texts separately")
    save_parents(TEST_PARENT_STORE_PATH, unique_parent_ids, unique_parent_texts)

    print("Embedding sample children.")
    embeddings = embed_texts(child_texts, settings.embedding_model)

    print("Adding children to Chroma ")
    add_children_to_store(collection, child_ids, child_texts, embeddings, parent_ids)

    count = get_collection_count(collection)
    print(f"Count after insert: {count}")
    print(f"Count matches expected (3): {count == 3}")

    print("\n Test query")
    query_embedding = embed_texts(["What is agentic AI?"], settings.embedding_model)[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    top_doc = results['documents'][0][0]
    top_parent_id = results['metadatas'][0][0]['parent_id']
    top_distance = results['distances'][0][0]

    print(f"Top match child doc : {top_doc}")
    print(f"Top match parent_id : {top_parent_id}")
    print(f"Top match distance  : {top_distance:.4f}")

    print("\n Fetch parent text using parent_id -")
    parent_text = get_parent_text(TEST_PARENT_STORE_PATH, top_parent_id)
    print(f"Parent text: {parent_text}")

    # remove test storage
    reset_collection(client, TEST_COLLECTION_NAME)
    reset_parent_store(TEST_PARENT_STORE_PATH)


if __name__ == "__main__":
    test_chroma_client()