from pathlib import Path
from pypdf import PdfReader


class PDFLoadError(Exception):
    pass


def load_pdf_text(pdf_path: str) -> str:
    path = Path(pdf_path)

    if not path.exists():
        raise PDFLoadError(
            f"PDF not found at '{pdf_path}'. "
        )

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise PDFLoadError(f"Failed to open PDF '{pdf_path}': {e}")

    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise PDFLoadError(
            f"PDF '{pdf_path}' was read but no extractable text was found "
        )

    return full_text


def get_page_count(pdf_path: str) -> int:
    reader = PdfReader(str(Path(pdf_path)))
    return len(reader.pages)