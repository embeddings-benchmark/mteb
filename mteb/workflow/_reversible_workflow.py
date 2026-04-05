from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WorkflowFailureError(RuntimeError):
    """Exception raised when a workflow fails with rollback status information.

    Attributes:
        step_name: Name of the step where the workflow failed.
        original_exception: The exception that caused the workflow to fail.
        failed_undo_steps: List of step names whose undo() calls failed during rollback.
    """

    def __init__(
        self,
        step_name: str,
        original_exception: Exception,
        failed_undo_steps: list[str] | None = None,
    ) -> None:
        """Initialize the exception with structured failure information.

        Args:
            step_name: Name of the step where the workflow failed.
            original_exception: The exception that triggered the rollback.
            failed_undo_steps: Optional list of step names that failed to undo.
        """
        self.step_name = step_name
        self.original_exception = original_exception
        self.failed_undo_steps = failed_undo_steps or []

        # Build message
        if self.failed_undo_steps:
            message = (
                f"Workflow failed at {step_name}. "
                f"Rollback attempted but {len(self.failed_undo_steps)} undo step(s) failed: "
                f"{', '.join(self.failed_undo_steps)}. Manual cleanup may be required."
            )
        else:
            message = (
                f"Workflow failed at {step_name}. "
                f"All completed steps have been rolled back."
            )

        super().__init__(message)


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
                 Steps should accept context via their constructor and store it as an instance
                 variable to access shared data. For example:
                 - step1 computes a value and stores in context["key"] = value
                 - step2 retrieves it via self.context["key"]

    Examples:
        >>> context = {}
        >>> step1 = MyAction(context=context)
        >>> step2 = OtherAction(context=context)
        >>> workflow = ReversibleWorkflow(steps=[step1, step2], context=context)
        >>> try:
        ...     workflow.run()
        ... except WorkflowFailureError as e:
        ...     print(f"Workflow failed at {e.step_name}: {e.original_exception}")
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
        - Tracks failed undo attempts and reports their status in the error message

        Raises:
            RuntimeError: Wraps the original exception with rollback status.
                         The __cause__ contains the original exception.
                         Message indicates whether rollback was fully successful or partially failed.
        """
        completed: list[ReversibleAction] = []
        failed_undo_steps: list[str] = []

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
                        failed_undo_steps.append(rollback_name)
                        logger.error(
                            f"Undo for {rollback_name} also failed: {undo_error}. "
                            f"Manual cleanup may be required."
                        )

                logger.error("Rollback completed. Raising WorkflowFailureError...")
                raise WorkflowFailureError(
                    step_name=step_name,
                    original_exception=e,
                    failed_undo_steps=failed_undo_steps,
                ) from e
