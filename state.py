from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    task: str
    model: str  # LiteLLM model id, e.g. gpt-4o, anthropic/claude-3-5-sonnet-20241022
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: list[dict]
    tokens_used: int
    max_tokens: int
    status: str  # "running" | "summarizing" | "complete"
    summary: str | None
