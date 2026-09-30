from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ParentChunk:
    id: str
    text: str
    chunk_index: int


@dataclass
class ChildChunk:
    id: str
    text: str
    chunk_index: int
    parent_id: str


def _split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)


def hierarchical_chunk_text(
    text: str,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 1000,
    child_chunk_overlap: int = 150,
) -> tuple[list[ParentChunk], list[ChildChunk]]:

    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    # split into parent chunks
    raw_parents = _split(text, parent_chunk_size, parent_chunk_overlap)
    parents = [
        ParentChunk(id=f"parent_{i}", text=raw_text, chunk_index=i)
        for i, raw_text in enumerate(raw_parents)
    ]

    # split each parent into child chunks
    children: list[ChildChunk] = []
    child_counter = 0
    for parent in parents:
        raw_children = _split(parent.text, child_chunk_size, child_chunk_overlap)
        for raw_child_text in raw_children:
            children.append(
                ChildChunk(
                    id=f"child_{child_counter}",
                    text=raw_child_text,
                    chunk_index=child_counter,
                    parent_id=parent.id,
                )
            )
            child_counter += 1

    return parents, children


def get_parent_text(parent_id: str, parents: list[ParentChunk]) -> str:
    for parent in parents:
        if parent.id == parent_id:
            return parent.text
    raise ValueError(f"No parent found with id '{parent_id}'")