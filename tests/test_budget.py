import pytest
from comptroller.budget import TokenBudget, count_tokens


def test_token_budget_starts_at_zero():
    """TokenBudget starts at 0."""
    budget = TokenBudget(max_tokens=1000)
    assert budget.used == 0


def test_record_increments_correctly():
    """record() increments correctly."""
    budget = TokenBudget(max_tokens=1000)
    budget.record(100)
    assert budget.used == 100
    budget.record(200)
    assert budget.used == 300


def test_is_exceeded_behavior():
    """is_exceeded() is False below limit, True at and above limit."""
    budget = TokenBudget(max_tokens=1000)

    # Below limit
    budget.record(500)
    assert budget.is_exceeded() is False

    # At limit
    budget.record(500)
    assert budget.is_exceeded() is True

    # Above limit
    budget.record(100)
    assert budget.is_exceeded() is True


def test_remaining_never_negative():
    """remaining() never goes below 0."""
    budget = TokenBudget(max_tokens=1000)

    budget.record(500)
    assert budget.remaining() == 500

    budget.record(500)
    assert budget.remaining() == 0

    budget.record(500)
    assert budget.remaining() == 0


def test_pct_used_calculation():
    """pct_used() returns correct float."""
    budget = TokenBudget(max_tokens=1000)

    budget.record(500)
    assert budget.pct_used() == 0.5

    budget.record(250)
    assert budget.pct_used() == 0.75

    budget.record(250)
    assert budget.pct_used() == 1.0


def test_count_tokens():
    """count_tokens returns a positive integer for non-empty text."""
    tokens = count_tokens("Hello, world!")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_pct_used_with_zero_max_tokens():
    """pct_used() handles zero max_tokens without division error."""
    budget = TokenBudget(max_tokens=0)
    budget.record(100)
    # Should return 1.0 instead of raising ZeroDivisionError
    assert budget.pct_used() == 1.0
