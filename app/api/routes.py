from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.vectorstore.chroma_client import get_client, get_or_create_collection
from app.graph.build_graph import build_rag_graph
from app.ingestion.ingest_pipeline import run_ingestion
from app.ingestion.pdf_loader import PDFLoadError
from fastapi import HTTPException


app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0")

_settings = get_settings()
_client = get_client(_settings.chroma_persist_dir)
_collection = get_or_create_collection(_client, _settings.chroma_collection_name)
_graph = build_rag_graph(_collection, _settings)


class ChatRequest(BaseModel):
    question: str


class RetrievedChunkResponse(BaseModel):
    parent_id: str
    text: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    confidence_score: float

class IngestResponse(BaseModel):
    pdf_path: str
    page_count: int
    parent_chunks: int
    child_chunks: int
    stored_count: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = _graph.invoke({"question": request.question})

    chunks = [
        RetrievedChunkResponse(
            parent_id=c["parent_id"],
            text=c["parent_text"],
            similarity_score=c["similarity_score"],
        )
        for c in result["retrieved_chunks"]
    ]

    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=chunks,
        confidence_score=result["confidence_score"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global _collection, _graph

    try:
        summary = run_ingestion(_settings)
    except PDFLoadError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _collection = get_or_create_collection(_client, _settings.chroma_collection_name)
    _graph = build_rag_graph(_collection, _settings)

    return IngestResponse(**summary)