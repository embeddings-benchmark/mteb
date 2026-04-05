from __future__ import annotations

import pytest

from mteb._reversible_workflow.reversible_workflow import (
    ReversibleWorkflow,
    WorkflowFailureError,
)


class _DummyAction:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def do(self) -> None:
        self.events.append(f"do:{self.name}")

    def undo(self) -> None:
        self.events.append(f"undo:{self.name}")


class _FailingAction(_DummyAction):
    def do(self) -> None:
        self.events.append(f"do:{self.name}")
        raise ValueError("boom")


class _UndoFailAction(_DummyAction):
    def undo(self) -> None:
        self.events.append(f"undo:{self.name}")
        raise RuntimeError("undo failed")


def test_reversible_workflow_runs_all_steps_when_successful() -> None:
    events: list[str] = []
    workflow = ReversibleWorkflow(
        steps=[_DummyAction(events, "a"), _DummyAction(events, "b")], context={}
    )

    workflow.run()

    assert events == ["do:a", "do:b"]


def test_reversible_workflow_rolls_back_completed_steps_in_reverse_order() -> None:
    events: list[str] = []
    workflow = ReversibleWorkflow(
        steps=[
            _DummyAction(events, "a"),
            _DummyAction(events, "b"),
            _FailingAction(events, "c"),
        ],
        context={},
    )

    with pytest.raises(WorkflowFailureError) as exc_info:
        workflow.run()

    exc = exc_info.value
    # Check structured exception attributes
    assert exc.step_name == "_FailingAction"
    assert isinstance(exc.original_exception, ValueError)
    assert str(exc.original_exception) == "boom"
    # Verify all steps were rolled back successfully (no failed undo steps)
    assert exc.failed_undo_steps == []

    assert events == ["do:a", "do:b", "do:c", "undo:b", "undo:a"]


def test_reversible_workflow_continues_rollback_when_undo_fails() -> None:
    events: list[str] = []
    workflow = ReversibleWorkflow(
        steps=[
            _DummyAction(events, "a"),
            _UndoFailAction(events, "b"),
            _FailingAction(events, "c"),
        ],
        context={},
    )

    with pytest.raises(WorkflowFailureError) as exc_info:
        workflow.run()

    exc = exc_info.value
    # Check structured exception attributes
    assert exc.step_name == "_FailingAction"
    # Verify that _UndoFailAction's undo failed
    assert "_UndoFailAction" in exc.failed_undo_steps
    assert len(exc.failed_undo_steps) == 1

    # Undo of b fails, but a is still undone.
    assert events == ["do:a", "do:b", "do:c", "undo:b", "undo:a"]
