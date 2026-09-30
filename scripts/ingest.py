import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion.ingest_pipeline import run_ingestion
from app.ingestion.pdf_loader import PDFLoadError


def main():
    settings = get_settings()
    print(f"Ingesting PDF from: {settings.pdf_local_path}")

    try:
        summary = run_ingestion(settings)
    except PDFLoadError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print(f"Done. Pages: {summary['page_count']} | "
          f"Parent chunks: {summary['parent_chunks']} | "
          f"Child chunks: {summary['child_chunks']} | "
          f"Stored: {summary['stored_count']}")


if __name__ == "__main__":
    main()