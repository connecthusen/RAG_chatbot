from app.config import get_settings
from app.ingestion.pdf_loader import load_pdf_text, get_page_count, PDFLoadError


def test_pdf_loader():
    settings = get_settings()
    pdf_path = settings.pdf_local_path

    print(f"\nLoading PDF from: {pdf_path}")

    try:
        page_count = get_page_count(pdf_path)
        print(f"Page count       : {page_count}")

        text = load_pdf_text(pdf_path)
        print(f"Total characters : {len(text)}")
        print(f"Total words      : {len(text.split())}")

        print("\n First 500 characters ")
        print(text[:500])
        print("---------------------------------\n")
    except PDFLoadError as e:
        print(f"PDF load failed: {e}")


if __name__ == "__main__":
    test_pdf_loader()