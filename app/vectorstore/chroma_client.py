import chromadb
from chromadb.config import Settings as ChromaSettings


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))


def get_or_create_collection(client: chromadb.ClientAPI, collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_children_to_store(
    collection,
    child_ids: list[str],
    child_texts: list[str],
    child_embeddings: list[list[float]],
    parent_ids: list[str],
) -> None:

    if not (len(child_ids) == len(child_texts) == len(child_embeddings) == len(parent_ids)):
        raise ValueError("All input lists must be the same length.")

    metadatas = [{"parent_id": pid} for pid in parent_ids]

    collection.upsert(
        ids=child_ids,
        embeddings=child_embeddings,
        documents=child_texts,
        metadatas=metadatas,
    )


def get_collection_count(collection) -> int:
    return collection.count()


def reset_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass