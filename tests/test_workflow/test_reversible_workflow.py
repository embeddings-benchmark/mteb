from __future__ import annotations

import pytest

from mteb._reversible_workflow.reversible_workflow import ( # noqa: PLC2701
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
    assert events == ["do:a", "do:b", "do:c", "undo:b", "undo:a"]
