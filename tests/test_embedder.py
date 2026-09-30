from app.config import get_settings
from app.embeddings.embedder import embed_texts, embed_query


def test_embedder():
    settings = get_settings()

    sample_texts = [
        "Agentic AI systems can plan and execute multi-step tasks autonomously.",
        "A retrieval-augmented generation pipeline combines search with an LLM.",
        "The weather today is sunny with a light breeze.",
    ]

    print(f"\nEmbedding model: {settings.embedding_model}")
    print(f"Expected dimension: {settings.embedding_dimension}")

    print("\nEmbedding sample texts.")
    embeddings = embed_texts(sample_texts, settings.embedding_model)

    print(f"Number of embeddings returned: {len(embeddings)}")
    print(f"Dimension of each embedding  : {len(embeddings[0])}")
    print(f"Dimension matches config     : {len(embeddings[0]) == settings.embedding_dimension}")

    print("\n First embedding (10 values)")
    print(embeddings[0][:10])

    print("\nEmbedding a query:What is agentic AI?")
    query_embedding = embed_query("What is agentic AI?", settings.embedding_model)
    print(f"Query embedding dimension: {len(query_embedding)}")


    import numpy as np
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("\nSimilarity check")
    for i, text in enumerate(sample_texts):
        sim = cosine_sim(query_embedding, embeddings[i])
        print(f"Sim to: \"{text[:50]}...\" -> {sim:.4f}")

if __name__ == "__main__":
    test_embedder()