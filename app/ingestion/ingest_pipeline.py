from app.ingestion.pdf_loader import load_pdf_text, get_page_count, PDFLoadError
from app.ingestion.chunker import hierarchical_chunk_text
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_client import (
    get_client,
    get_or_create_collection,
    add_children_to_store,
    get_collection_count,
    reset_collection,
)
from app.vectorstore.parent_store import save_parents, reset_parent_store


def run_ingestion(settings) -> dict:

    #  Load PDF
    page_count = get_page_count(settings.pdf_local_path)
    text = load_pdf_text(settings.pdf_local_path)

    # Hierarchical chunking
    parents, children = hierarchical_chunk_text(
        text,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        child_chunk_size=settings.chunk_size,
        child_chunk_overlap=settings.chunk_overlap,
    )

    # Embed child chunks
    child_texts = [c.text for c in children]
    embeddings = embed_texts(child_texts, settings.embedding_model)

    # Reset and stores
    client = get_client(settings.chroma_persist_dir)
    reset_collection(client, settings.chroma_collection_name)
    reset_parent_store(settings.parent_store_path)
    collection = get_or_create_collection(client, settings.chroma_collection_name)

    parent_ids_for_save = [p.id for p in parents]
    parent_texts_for_save = [p.text for p in parents]
    save_parents(settings.parent_store_path, parent_ids_for_save, parent_texts_for_save)

    child_ids = [c.id for c in children]
    child_parent_ids = [c.parent_id for c in children]
    add_children_to_store(collection, child_ids, child_texts, embeddings, child_parent_ids)

    final_count = get_collection_count(collection)

    return {
        "pdf_path": settings.pdf_local_path,
        "page_count": page_count,
        "parent_chunks": len(parents),
        "child_chunks": len(children),
        "stored_count": final_count,
    }