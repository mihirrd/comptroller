from typing import Any

import tiktoken

import litellm
from litellm import completion_cost


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


def _coerce_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _usage_openai_dict_from_message(msg: Any) -> dict[str, int]:
    """Map LangChain ``usage_metadata`` (input/output tokens) to OpenAI-style usage keys."""
    um = getattr(msg, "usage_metadata", None)
    if um is None:
        return {}
    getter = um.get if isinstance(um, dict) else lambda k, default=None: getattr(um, k, default)
    out: dict[str, int] = {}
    inp = _coerce_int(getter("input_tokens"))
    out_tok = _coerce_int(getter("output_tokens"))
    total = _coerce_int(getter("total_tokens"))
    if inp is not None:
        out["prompt_tokens"] = inp
    if out_tok is not None:
        out["completion_tokens"] = out_tok
    if total is not None:
        out["total_tokens"] = total
    return out


def _model_name_for_cost(msg: Any, fallback_model: str) -> str:
    rm = getattr(msg, "response_metadata", None) or {}
    if isinstance(rm, dict):
        mn = rm.get("model_name")
        if mn:
            return str(mn)
    return fallback_model


def dollars_from_llm_message(msg: Any, model: str) -> float:
    """Estimated USD cost for one chat completion using LiteLLM's pricing tables.

    Prefer ``litellm.completion_cost`` with token splits from provider ``usage_metadata``.
    If only ``total_tokens`` is present, approximate with ``litellm.get_model_info`` rates
    (conservative: multiply by the larger of input/output per-token prices).
    """
    model_name = _model_name_for_cost(msg, model)
    usage = _usage_openai_dict_from_message(msg)

    litellm_usage: dict[str, int] = {}
    if "prompt_tokens" in usage:
        litellm_usage["prompt_tokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        litellm_usage["completion_tokens"] = usage["completion_tokens"]

    if litellm_usage:
        try:
            cost = completion_cost(
                completion_response={"model": model_name, "usage": litellm_usage},
                model=model_name,
            )
        except Exception:
            cost = 0.0
        return float(cost)

    total = usage.get("total_tokens")
    if total is None or total <= 0:
        return 0.0

    try:
        info = litellm.get_model_info(model_name)
    except Exception:
        return 0.0

    in_c = float(info.get("input_cost_per_token") or 0.0)
    out_c = float(info.get("output_cost_per_token") or 0.0)
    if in_c <= 0.0 and out_c <= 0.0:
        return 0.0
    # Unknown input/output split: upper-bound using the higher per-token rate.
    return float(total) * max(in_c, out_c)
