import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(
            f"Missing required environment variable: '{key}'. "
            f"Check your .env file "
        )
    return value


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: _get_env("GROQ_API_KEY", required=True))
    llm_model: str = field(default_factory=lambda: _get_env("LLM_MODEL", "llama-3.3-70b-versatile"))

    embedding_model: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dimension: int = field(default_factory=lambda: int(_get_env("EMBEDDING_DIMENSION", "384")))


    chroma_persist_dir: str = field(default_factory=lambda: _get_env("CHROMA_PERSIST_DIR", "data/vector_db"))
    chroma_collection_name: str = field(
        default_factory=lambda: _get_env("CHROMA_COLLECTION_NAME", "agentic_ai_ebook")
    )


    chunk_size: int = field(default_factory=lambda: int(_get_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_get_env("CHUNK_OVERLAP", "150")))


    top_k: int = field(default_factory=lambda: int(_get_env("TOP_K", "4")))
    similarity_threshold: float = field(default_factory=lambda: float(_get_env("SIMILARITY_THRESHOLD", "0.3")))


    pdf_local_path: str = field(default_factory=lambda: _get_env("PDF_LOCAL_PATH", "data/raw/agentic-ai-ebook.pdf"))


    api_host: str = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_get_env("API_PORT", "8000")))


def get_settings() -> Settings:
    return Settings()