"""Reversible workflow infrastructure for managing sequences of reversible actions with automatic rollback."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ReversibleAction(Protocol):
    """Protocol for actions that can be executed and undone.

    Any action used in ReversibleWorkflow must implement the do() and undo() methods.
    The undo() method should reverse the effects of do() as much as possible.
    """

    def do(self) -> None:
        """Execute the action.

        Raises:
            Exception: If the action fails. The exception will trigger rollback of completed steps.
        """
        ...

    def undo(self) -> None:
        """Undo/reverse the action after a successful do().

        This is called during rollback if a later step fails. Should restore the state
        to what it was before do() was called.

        Raises:
            Exception: If undo fails. Logged but doesn't prevent undoing other steps.
        """
        ...


class ReversibleWorkflow:
    """Orchestrator for executing a sequence of reversible actions with automatic rollback.

    Executes each step's do() method. If any step fails, automatically calls undo()
    on all previously completed steps in reverse order before re-raising the exception.

    Attributes:
        context: Shared mutable dict that steps can use to pass data to other steps.
                 For example: step1 computes commit_sha and stores in context["commit_sha"],
                 step2 retrieves it via self.context["commit_sha"].

    Examples:
        >>> workflow = ReversibleWorkflow(
        ...     steps=[step1, step2, step3],
        ...     context={}
        ... )
        >>> try:
        ...     workflow.run()
        ... except RuntimeError as e:
        ...     print(f"Workflow failed and rolled back: {e}")
    """

    def __init__(
        self,
        steps: list[ReversibleAction],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the workflow.

        Args:
            steps: List of ReversibleAction objects to execute in order.
            context: Optional shared context dict for steps to communicate. If None, empty dict is created.
        """
        self.steps = steps
        self.context = context or {}

    def run(self) -> None:
        """Execute all steps with automatic rollback on failure.

        - Executes each step's do() method in order
        - Tracks completed steps
        - On any failure, calls undo() on completed steps in reverse order
        - Re-raises the original exception after rollback attempts

        Raises:
            RuntimeError: Always wraps the original exception with rollback summary.
                         The __cause__ contains the original exception.
        """
        completed: list[ReversibleAction] = []

        for step in self.steps:
            step_name = type(step).__name__
            try:
                logger.debug(f"Executing {step_name}...")
                step.do()
                completed.append(step)
                logger.debug(f"{step_name} completed successfully")
            except Exception as e:
                logger.error(f"Failed at {step_name}: {e}")
                logger.debug(f"Rolling back {len(completed)} completed step(s)...")

                # Rollback in reverse order
                for rollback_step in reversed(completed):
                    rollback_name = type(rollback_step).__name__
                    try:
                        logger.debug(f"Undoing {rollback_name}...")
                        rollback_step.undo()
                        logger.debug(f"{rollback_name} undo completed")
                    except Exception as undo_error:
                        logger.error(
                            f"Undo for {rollback_name} also failed: {undo_error}. "
                            f"Manual cleanup may be required."
                        )

                logger.error("Rollback completed. Raising exception...")
                raise RuntimeError(
                    f"Workflow failed at {step_name}. All completed steps have been rolled back."
                ) from e
