from collections.abc import Callable

import pytest

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from tests.mock_tasks import MockRetrievalTask


def _meta(prompt: dict[str, str] | str | None) -> TaskMetadata:
    meta = MockRetrievalTask.metadata
    meta.prompt = prompt
    return meta


class _FakeEncoder(AbsEncoder):
    def __init__(
        self, instruction_template: str | Callable[[str, PromptType | None], str] | None
    ) -> None:
        self.instruction_template = instruction_template

    def encode(self, *args, **kwargs):
        raise NotImplementedError


def _gritlm_template(instr, prompt_type):
    return f"<|user|>\n{instr}\n<|embed|>\n" if instr else "<|embed|>\n"


QUERY_INSTR = "Given a biology post, retrieve relevant passages"


@pytest.mark.parametrize(
    "template, prompt_type, expected",
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
