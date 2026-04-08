"""LiteLLM-backed cost estimation from chat message metadata."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from comptroller.budget import dollars_from_llm_message


def test_dollars_from_usage_metadata_openai_style():
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        },
        response_metadata={"model_name": "gpt-4o"},
    )
    cost = dollars_from_llm_message(msg, "gpt-4o")
    assert cost == 0.0075


def test_dollars_total_tokens_fallback_uses_model_info():
    """When only total_tokens is present, use conservative per-token bound from LiteLLM model info."""
    msg = SimpleNamespace(
        usage_metadata={"total_tokens": 1000},
        response_metadata={"model_name": "gpt-4o"},
    )
    cost = dollars_from_llm_message(msg, "gpt-4o")
    assert cost > 0
