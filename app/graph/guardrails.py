import re
from dataclasses import dataclass

# prompt injection
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"disregard (all |the )?(previous|above|prior)",
    r"you are now",
    r"forget (all |the )?(previous|your) (instructions|rules|training)",
    r"system prompt",
    r"reveal your (instructions|prompt|rules)",
    r"act as (if you|a|an)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
]

#  sensitive/internal information
SENSITIVE_INFO_PATTERNS = [
    r"api[_\s]?key",
    r"secret[_\s]?key",
    r"access[_\s]?token",
    r"env(ironment)?[_\s]?variable",
    r"\.env\b",
    r"password",
    r"credentials",
    r"groq[_\s]?api",
]

# Basic profanity
BASIC_PROFANITY = [
    "fuck", "shit", "bitch", "asshole", "bastard",
]


@dataclass
class GuardrailResult:
    is_safe: bool
    reason: str | None = None  # why it was blocked, if applicable


def _matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def check_guardrails(question: str) -> GuardrailResult:

    if not question or not question.strip():
        return GuardrailResult(is_safe=False, reason="empty_input")

    if _matches_any(question, INJECTION_PATTERNS):
        return GuardrailResult(is_safe=False, reason="prompt_injection")

    if _matches_any(question, SENSITIVE_INFO_PATTERNS):
        return GuardrailResult(is_safe=False, reason="sensitive_info_request")

    question_lower = question.lower()
    if any(word in question_lower.split() or word in question_lower for word in BASIC_PROFANITY):
        return GuardrailResult(is_safe=False, reason="profanity")

    return GuardrailResult(is_safe=True)