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
