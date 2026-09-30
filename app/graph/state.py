
from typing import TypedDict


class RAGState(TypedDict):
    question: str
    is_safe: bool
    block_reason: str | None
    retrieved_chunks: list[dict]
    confidence_score: float
    answer: str