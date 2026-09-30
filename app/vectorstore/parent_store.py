import json
from pathlib import Path


def save_parents(parent_store_path: str, parent_ids: list[str], parent_texts: list[str]) -> None:
    path = Path(parent_store_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_all_parents(parent_store_path)
    existing.update(dict(zip(parent_ids, parent_texts)))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_all_parents(parent_store_path: str) -> dict[str, str]:
    path = Path(parent_store_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parent_text(parent_store_path: str, parent_id: str) -> str:
    parents = load_all_parents(parent_store_path)
    if parent_id not in parents:
        raise ValueError(f"No parent found with id '{parent_id}' in {parent_store_path}")
    return parents[parent_id]


def get_parent_texts(parent_store_path: str, parent_ids: list[str]) -> list[str]:
    parents = load_all_parents(parent_store_path)
    return [parents[pid] for pid in parent_ids if pid in parents]


def reset_parent_store(parent_store_path: str) -> None:
    path = Path(parent_store_path)
    if path.exists():
        path.unlink()