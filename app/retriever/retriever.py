import math
from dataclasses import dataclass

from app.embeddings.embedder import embed_query
from app.vectorstore.parent_store import get_parent_text
from app.retriever.reranker import rerank


@dataclass
class RetrievedContext:
    parent_id: str
    parent_text: str
    similarity_score: float  # normalized 0-1 rerank score, higher is better
    matched_child_text: str  # the specific child snippet that triggered this match


def _normalize_rerank_score(score: float) -> float:

    return 1 / (1 + math.exp(-score))


def retrieve(
    question: str,
    collection,
    embedding_model: str,
    parent_store_path: str,
    rerank_model: str,
    retrieval_candidates: int = 10,
    top_k: int = 4,
    hf_token: str = "",
) -> list[RetrievedContext]:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # wide dense search
    query_embedding = embed_query(question, embedding_model)
    results = collection.query(query_embeddings=[query_embedding], n_results=retrieval_candidates)

    child_docs = results["documents"][0]
    child_metadatas = results["metadatas"][0]

    if not child_docs:
        return []

    # rerank candidates with cross-encoder
    reranked = rerank(question, child_docs, rerank_model, top_k=top_k, hf_token=hf_token)

    # resolve parents, de-duplicate
    seen_parent_ids: set[str] = set()
    contexts: list[RetrievedContext] = []

    for idx, score in reranked:
        parent_id = child_metadatas[idx]["parent_id"]

        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        parent_text = get_parent_text(parent_store_path, parent_id)
        normalized_score = _normalize_rerank_score(score)

        contexts.append(
            RetrievedContext(
                parent_id=parent_id,
                parent_text=parent_text,
                similarity_score=normalized_score,
                matched_child_text=child_docs[idx],
            )
        )

    return contexts 