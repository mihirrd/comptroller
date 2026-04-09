from typing import TypedDict, Annotated, NotRequired
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    task: str
    model: str  # LiteLLM model id, e.g. gpt-4o, anthropic/claude-3-5-sonnet-20241022
    workspace_root: str  # process cwd or explicit project root for prompts / tools
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: list[dict]
    tokens_used: int
    max_tokens: int
    api_dollars_used: float  # cumulative estimated API spend (USD) from LiteLLM pricing
    max_api_dollars: float | None  # None = no dollar cap (token budget may still apply)
    wall_seconds_used: float  # cumulative wall time (monotonic) while agent/tools/summarize run
    max_wall_seconds: float | None  # None = no wall-time cap
    session_retries_used: int  # LLM transient-error backoff retries consumed this session
    max_session_retries: int | None  # None = unlimited (per-call COMPTROLLER_LLM_RETRY_ATTEMPTS still applies)
    status: str  # "running" | "summarizing" | "complete"
    summary: str | None
    recent_files: NotRequired[list[str]]  # paths touched by tools; shown via per-turn context message, not system prompt
