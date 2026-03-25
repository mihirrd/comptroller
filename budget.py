from typing import Any

import tiktoken


class TokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0

    def record(self, tokens: int):
        self.used += tokens

    def is_exceeded(self) -> bool:
        return self.used >= self.max_tokens

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def pct_used(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self.used / self.max_tokens


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def tokens_from_llm_message(msg: Any, model: str = "gpt-4o") -> int:
    """Total tokens for one LLM API call from provider metadata when present; else tiktoken on message text.

    LangChain/LiteLLM populate ``usage_metadata`` (preferred) and often ``response_metadata['token_usage']``.
    """
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        t = um.get("total_tokens")
        if t is not None:
            return int(t)

    rm = getattr(msg, "response_metadata", None) or {}
    if isinstance(rm, dict):
        tu = rm.get("token_usage")
        if tu is not None:
            if isinstance(tu, dict):
                t = tu.get("total_tokens")
                if t is not None:
                    return int(t)
            t = getattr(tu, "total_tokens", None)
            if t is not None:
                return int(t)

    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        content = str(content)
    return count_tokens(str(content), model=model)
