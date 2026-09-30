from app.retriever.retriever import retrieve
from app.llm.groq_client import get_groq_client, generate_completion
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.graph.state import RAGState
from app.graph.guardrails import check_guardrails

NO_ANSWER_MESSAGE = "I cannot find this information in the provided document."

BLOCK_MESSAGES = {
    "empty_input": "Please enter a question.",
    "prompt_injection": "I can't process that request. Please ask a direct question about the document.",
    "sensitive_info_request": "I can't share system configuration, credentials, or internal information.",
    "profanity": "Please rephrase your question respectfully.",
}

def guardrail_node(state: RAGState) -> RAGState:
    result = check_guardrails(state["question"])
    return {
        **state,
        "is_safe": result.is_safe,
        "block_reason": result.reason,
    }

def route_after_guardrail(state: RAGState) -> str:
    return "retrieve" if state["is_safe"] else "refuse"


def refuse_node(state: RAGState) -> RAGState:
    message = BLOCK_MESSAGES.get(state["block_reason"], "I can't help with that request.")
    return {
        **state,
        "retrieved_chunks": [],
        "confidence_score": 0.0,
        "answer": message,
    }




def make_retrieve_node(collection, settings):

    def retrieve_node(state: RAGState) -> RAGState:
        contexts = retrieve(
            question=state["question"],
            collection=collection,
            embedding_model=settings.embedding_model,
            parent_store_path=settings.parent_store_path,
            rerank_model=settings.rerank_model,
            retrieval_candidates=settings.retrieval_candidates,
            top_k=settings.top_k,
            hf_token=settings.hf_token,
        )

        retrieved_chunks = [
            {
                "parent_id": c.parent_id,
                "parent_text": c.parent_text,
                "similarity_score": c.similarity_score,
                "matched_child_text": c.matched_child_text,
            }
            for c in contexts
        ]

        confidence_score = contexts[0].similarity_score if contexts else 0.0

        return {
            **state,
            "retrieved_chunks": retrieved_chunks,
            "confidence_score": confidence_score,
        }

    return retrieve_node


def make_generate_node(settings):

    client = get_groq_client(settings.groq_api_key)

    def generate_node(state: RAGState) -> RAGState:
        retrieved_chunks = state["retrieved_chunks"]

        if not retrieved_chunks or state["confidence_score"] < settings.similarity_threshold:
            return {**state, "answer": NO_ANSWER_MESSAGE}

        context_texts = [chunk["parent_text"] for chunk in retrieved_chunks]
        user_prompt = build_user_prompt(state["question"], context_texts)

        answer = generate_completion(
            client=client,
            model=settings.llm_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return {**state, "answer": answer}

    return generate_node