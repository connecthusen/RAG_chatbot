SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context from a document.

Rules you MUST follow:
1. Only use information found in the provided context to answer.
2. If the context does not contain enough information to answer the question, respond exactly with: "I cannot find this information in the provided document."
3. Do not use any outside knowledge, even if you know the answer.
4. Do not make assumptions or infer information not explicitly stated in the context.
5. Keep answers clear and concise, citing relevant details from the context.
6. Write your answer as clean, natural prose. Do NOT include labels like "(Context 1)", "(Context 2)", "[Context N]", or any other reference to the numbered context blocks — just answer naturally as if you already knew the information.
"""


def build_user_prompt(question: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    return f"""Context from document:

{context_block}

---

Question: {question}

Answer based only on the context above. Do not mention "Context 1", "Context 2", or any block numbers in your answer — just write the answer naturally:"""