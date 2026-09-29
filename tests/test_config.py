import os
from app.config import get_settings

def test_config():
    settings = get_settings()

    print("\nLoaded Settings ")
    print(f"GROQ_API_KEY         : {settings.groq_api_key}")
    print(f"LLM_MODEL             : {settings.llm_model}")
    print(f"EMBEDDING_MODEL       : {settings.embedding_model}")
    print(f"EMBEDDING_DIMENSION   : {settings.embedding_dimension}")
    print(f"CHROMA_PERSIST_DIR    : {settings.chroma_persist_dir}")
    print(f"CHROMA_COLLECTION     : {settings.chroma_collection_name}")
    print(f"CHUNK_SIZE            : {settings.chunk_size}")
    print(f"CHUNK_OVERLAP         : {settings.chunk_overlap}")
    print(f"TOP_K                 : {settings.top_k}")
    print(f"SIMILARITY_THRESHOLD  : {settings.similarity_threshold}")
    print(f"PDF_LOCAL_PATH        : {settings.pdf_local_path}")
    print(f"API_HOST:PORT         : {settings.api_host}:{settings.api_port}")
    print("----------------------------\n")

    print("✅ Config loaded successfully.")

if __name__ == "__main__":
    test_config()