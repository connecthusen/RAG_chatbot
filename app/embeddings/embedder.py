from sentence_transformers import SentenceTransformer

_model_cache: dict[str, SentenceTransformer] = {}

def _get_model(model_name: str, hf_token: str = "") -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name, token=hf_token or None)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = _get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str) -> list[float]:
    return embed_texts([query], model_name)[0]