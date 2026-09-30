from sentence_transformers import CrossEncoder

_reranker_cache: dict[str, CrossEncoder] = {}


def _get_reranker(model_name: str, hf_token: str = "") -> CrossEncoder:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(model_name, token=hf_token or None)
    return _reranker_cache[model_name]


def rerank(
    query: str,
    candidate_texts: list[str],
    model_name: str,
    top_k: int,
    hf_token: str = "",
) -> list[tuple[int, float]]:

    if not candidate_texts:
        return []

    model = _get_reranker(model_name, hf_token)
    pairs = [[query, text] for text in candidate_texts]
    scores = model.predict(pairs)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return [(idx, float(score)) for idx, score in indexed_scores[:top_k]]