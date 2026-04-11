from comptroller_swebench.prompt import build_task_prompt


def test_build_task_prompt_includes_problem_and_workspace():
    row = {
        "instance_id": "org__repo-1",
        "problem_statement": "Fix the bug.",
        "hints_text": "",
    }
    out = build_task_prompt(row, workspace_root="/work/repo")
    assert "org__repo-1" in out
    assert "Fix the bug." in out
    assert "/work/repo" in out


def test_build_task_prompt_includes_hints_when_present():
    row = {
        "instance_id": "x",
        "problem_statement": "P",
        "hints_text": "Try foo.",
    }
    out = build_task_prompt(row, workspace_root="/r")
    assert "Try foo." in out
