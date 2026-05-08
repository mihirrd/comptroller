"""Integration tests for graph budget policies: summarize, degrade, and stop+extend."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from comptroller.runner import AgentTurnInput, run_agent_turn


def test_policy_summarize_path(llm_scenario, tmp_path: Path):
    """Token cap without interactive or fallback should summarize and complete."""
    llm_scenario(
        agent=[AIMessage(content="burn budget quickly for summarize path")],
        summary=[AIMessage(content="summary: token cap reached")],
    )

    out = run_agent_turn(
        AgentTurnInput(
            task="summarize me",
            session_id="itg-summarize",
            max_tokens=1,
        ),
        db_path=str(tmp_path / "summarize.db"),
        log_steps_to_store=False,
        stream_chunk_budget_matches_max_tokens=False,
    )

    assert out.final_state is not None
    assert out.final_state["status"] == "complete"
    assert out.final_state["summary"] == "summary: token cap reached"


def test_policy_degrade_path(llm_scenario, tmp_path: Path):
    """Token cap with local fallback should degrade model and keep running."""
    llm_scenario(agent=[AIMessage(content="burn budget to force degradation")])

    out = run_agent_turn(
        AgentTurnInput(
            task="degrade me",
            session_id="itg-degrade",
            max_tokens=1,
            local_model_url="http://127.0.0.1:11434/v1",
            local_model_id="llama3.2",
            model_degraded=False,
        ),
        db_path=str(tmp_path / "degrade.db"),
        log_steps_to_store=False,
        stream_chunk_budget_matches_max_tokens=False,
    )

    assert out.final_state is not None
    assert out.final_state["status"] == "running"
    assert out.final_state["model_degraded"] is True
    assert out.final_state["model"] == "llama3.2"
    assert out.final_state["local_model_url"] == "http://127.0.0.1:11434/v1"


def test_policy_stop_then_extend_path(llm_scenario, tmp_path: Path):
    """Interactive budget should stop at awaiting_budget, then continue after increasing caps."""
    llm_scenario(
        agent=[
            AIMessage(content="first pass exceeds budget"),
            AIMessage(content="second pass after extend continues"),
        ]
    )
    db_path = str(tmp_path / "extend.db")
    session_id = "itg-extend"

    first = run_agent_turn(
        AgentTurnInput(
            task="first turn",
            session_id=session_id,
            max_tokens=1,
            interactive_budget=True,
        ),
        db_path=db_path,
        log_steps_to_store=False,
        stream_chunk_budget_matches_max_tokens=False,
    )
    assert first.final_state is not None
    assert first.final_state["status"] == "awaiting_budget"

    extended_max_tokens = int(first.final_state["max_tokens"]) + 100
    second = run_agent_turn(
        AgentTurnInput(
            task="extend and continue",
            session_id=session_id,
            tokens_used=int(first.final_state["tokens_used"]),
            max_tokens=extended_max_tokens,
            api_dollars_used=float(first.final_state.get("api_dollars_used") or 0.0),
            max_api_dollars=first.final_state.get("max_api_dollars"),
            wall_seconds_used=float(first.final_state.get("wall_seconds_used") or 0.0),
            max_wall_seconds=first.final_state.get("max_wall_seconds"),
            session_retries_used=int(first.final_state.get("session_retries_used") or 0),
            interactive_budget=True,
            append_user_message=False,
        ),
        db_path=db_path,
        log_steps_to_store=False,
        stream_chunk_budget_matches_max_tokens=False,
    )

    assert second.final_state is not None
    assert second.final_state["status"] == "running"
    assert second.final_state["tokens_used"] > int(first.final_state["tokens_used"])
