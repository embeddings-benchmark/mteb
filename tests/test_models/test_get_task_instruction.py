from collections.abc import Callable
from typing import Any

import pytest

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.abs_encoder import AbsEncoder, PromptSource
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.types import PromptType


def _meta(prompt: dict[str, str] | str | None) -> TaskMetadata:
    meta = MockRetrievalTask.metadata
    meta.prompt = prompt
    return meta


class _FakeEncoder(AbsEncoder):
    def __init__(
        self, instruction_template: str | Callable[[str, PromptType | None], str] | None
    ) -> None:
        self.instruction_template = instruction_template

    def encode(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


def _gritlm_template(instr, prompt_type):
    return f"<|user|>\n{instr}\n<|embed|>\n" if instr else "<|embed|>\n"


QUERY_INSTR = "Given a biology post, retrieve relevant passages"


@pytest.mark.parametrize(
    ("template", "prompt_type", "expected"),
    [
        # Callable template: fires even on empty input (document side)
        (_gritlm_template, PromptType.document, "<|embed|>\n"),
        # Callable template: renders normally on non-empty input (query side)
        (_gritlm_template, PromptType.query, f"<|user|>\n{QUERY_INSTR}\n<|embed|>\n"),
        # String template: gated on empty input (document side) → falsy
        ("Instruct: {instruction}\nQuery: ", PromptType.document, ""),
        # String template: renders on non-empty input (query side)
        (
            "Instruct: {instruction}\nQuery: ",
            PromptType.query,
            f"Instruct: {QUERY_INSTR}\nQuery: ",
        ),
        # No template: returns raw instruction (empty for document side)
        (None, PromptType.document, ""),
        # No template: returns raw instruction (query side)
        (None, PromptType.query, QUERY_INSTR),
    ],
)
def test_get_task_instruction(template, prompt_type, expected):
    """Regression test for issue https://github.com/embeddings-benchmark/mteb/issues/4683"""
    enc = _FakeEncoder(instruction_template=template)
    meta = _meta(prompt={"query": QUERY_INSTR})
    result = enc.get_task_instruction(meta, prompt_type)
    assert (result or "") == expected


TASK_FIRST = (
    PromptSource.CUSTOM_PROMPT,
    PromptSource.DEFAULT_TASK_PROMPT,
    PromptSource.TASK_TYPE_PROMPT,
    PromptSource.ENCODE_TYPE_PROMPT,
)


@pytest.mark.parametrize(
    ("prompts_dict", "prompt_priority", "prompt_type", "expected"),
    [
        # default order: generic model prompts override the task prompt
        (
            {"Retrieval": "task type", "query": "query"},
            None,
            PromptType.query,
            "task type",
        ),
        # task first: task prompt overrides generic model prompts
        (
            {"Retrieval": "task type", "query": "query"},
            TASK_FIRST,
            PromptType.query,
            QUERY_INSTR,
        ),
        # task first: task name prompt from the model is still used first
        ({"MockRetrievalTask": "custom"}, TASK_FIRST, PromptType.query, "custom"),
        # task first: falls back to model prompts when the task has no prompt for the prompt type
        ({"document": "document"}, TASK_FIRST, PromptType.document, "document"),
    ],
)
def test_get_instruction_prompt_priority(
    prompts_dict, prompt_priority, prompt_type, expected
):
    enc = _FakeEncoder(instruction_template=None)
    enc.prompts_dict = prompts_dict
    if prompt_priority:
        enc.prompt_priority = prompt_priority
    meta = _meta(prompt={"query": QUERY_INSTR})
    assert enc.get_instruction(meta, prompt_type) == expected


def test_instruct_wrapper_invalid_prompt_priority():
    # validated before the model is loaded
    with pytest.raises(ValueError, match="not a valid PromptSource"):
        InstructSentenceTransformerModel(
            "unused", revision="unused", prompt_priority=["custom_task"]
        )
