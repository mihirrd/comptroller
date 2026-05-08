"""Integration tests for compiled graph execution and checkpoint resume flow."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from comptroller import store
from comptroller.graph import run_graph
from comptroller.runner import AgentTurnInput, run_agent_turn


class _FakeBoundLLM:
    def invoke(self, _messages):
        return AIMessage(content="integration-ok")


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, _tools):
        return _FakeBoundLLM()

    def invoke(self, _messages):
        return AIMessage(content="integration-summary")


def test_run_graph_end_to_end_with_compiled_graph(monkeypatch, tmp_path: Path):
    """run_graph should compile + stream through LangGraph without mocked graph object."""
    db_path = str(tmp_path / "graph.db")
    monkeypatch.setattr("comptroller.graph.ChatLiteLLM", _FakeLLM)

    out = run_graph(
        task="say hi",
        max_tokens=200,
        session_id="itg-graph-1",
        model="gpt-4o",
        db_path=db_path,
    )

    assert out is not None
    assert out["session_id"] == "itg-graph-1"
    assert out["status"] in {"running", "complete"}
    assert out["tokens_used"] >= 0
    assert out["messages"]


def test_agent_checkpoint_resume_integration(monkeypatch, tmp_path: Path):
    """LangGraph checkpoint ids from run_agent_turn should round-trip via store pending resume fields."""
    db_path = str(tmp_path / "checkpoint.db")
    store.init_db(db_path)
    store.create_session("itg-resume-1", "task", 200, db_path)
    monkeypatch.setattr("comptroller.graph.ChatLiteLLM", _FakeLLM)

    first = run_agent_turn(
        AgentTurnInput(task="first turn", session_id="itg-resume-1", max_tokens=200),
        db_path=db_path,
        log_steps_to_store=False,
    )

    assert first.langgraph_checkpoint_id
    ns = first.langgraph_checkpoint_ns or ""
    store.set_session_pending_resume("itg-resume-1", first.langgraph_checkpoint_id, ns, db_path)
    assert store.get_session_pending_resume("itg-resume-1", db_path) == (
        first.langgraph_checkpoint_id,
        ns,
    )

    resumed = run_agent_turn(
        AgentTurnInput(
            task="resume",
            session_id="itg-resume-1",
            max_tokens=200,
            append_user_message=False,
            resume_langgraph_checkpoint_id=first.langgraph_checkpoint_id,
            resume_langgraph_checkpoint_ns=ns,
        ),
        db_path=db_path,
        log_steps_to_store=False,
    )
    assert resumed.final_state is not None
    assert resumed.langgraph_checkpoint_id is not None
