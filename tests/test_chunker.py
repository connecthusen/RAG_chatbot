from app.config import get_settings
from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.chunker import hierarchical_chunk_text, get_parent_text


def test_chunker():
    settings = get_settings()

    print(f"\nLoading PDF from: {settings.pdf_local_path}")
    text = load_pdf_text(settings.pdf_local_path)
    print(f"Total characters loaded: {len(text)}")

    parents, children = hierarchical_chunk_text(
        text,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        child_chunk_size=settings.chunk_size,
        child_chunk_overlap=settings.chunk_overlap,
    )

    print(f"\nParent chunk size/overlap: {settings.parent_chunk_size}/{settings.parent_chunk_overlap}")
    print(f"Child chunk size/overlap : {settings.chunk_size}/{settings.chunk_overlap}")
    print(f"Total parent chunks      : {len(parents)}")
    print(f"Total child chunks       : {len(children)}")
    print(f"Avg children per parent  : {len(children) / len(parents):.1f}")

    print("\nFirst parent ")
    print(f"ID    : {parents[0].id}")
    print(f"Length: {len(parents[0].text)} chars")
    print(f"Text  : {parents[0].text[:300]}...")

    print("\n First child (first parent)")
    first_child = children[0]
    print(f"ID        : {first_child.id}")
    print(f"Parent ID : {first_child.parent_id}")
    print(f"Length    : {len(first_child.text)} chars")
    print(f"Text      : {first_child.text[:200]}...")

    print("\nVerify parent lookup works ")
    looked_up = get_parent_text(first_child.parent_id, parents)
    print(f"Lookup successful: {looked_up == parents[0].text}")


if __name__ == "__main__":
    test_chunker()