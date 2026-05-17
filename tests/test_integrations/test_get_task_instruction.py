"""Regression tests for `AbsEncoder.get_task_instruction` empty-instruction
routing. Callable templates must be invoked even when the task does not
specify an instruction body for the given prompt_type (typical for
documents under tasks with `prompt={"query": "..."}` only), otherwise
models like ReasonIR/GritLM whose templates emit a non-empty prefix on
empty input (e.g. `"<|embed|>\n"`) end up encoding documents without that
prefix. See https://github.com/embeddings-benchmark/mteb/issues/<TBD>.
"""

from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType


def _meta(prompt: dict[str, str] | str | None) -> TaskMetadata:
    return TaskMetadata(
        name="MockBrightProBiologyRetrieval",
        dataset={"path": "mock/Bright-Pro", "revision": "abc"},
        reference="https://example.com",
        description="mock task for unit-testing empty-instruction routing",
        type="Retrieval",
        prompt=prompt,
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-01", "2026-01-02"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation="",
    )


class _FakeEncoder(AbsEncoder):
    """Minimal concrete AbsEncoder for unit-testing get_task_instruction.

    AbsEncoder is abstract; we provide trivial overrides so it instantiates.
    """

    def __init__(self, instruction_template):
        self.instruction_template = instruction_template
        self.prompts_dict = None
        self.model_prompts = None
        self.model = None
        self.mteb_model_meta = None

    # required by AbsEncoder.ABC
    def encode(self, *args, **kwargs):  # pragma: no cover - not called in tests
        raise NotImplementedError


def test_callable_template_invoked_for_empty_document_instruction():
    """ReasonIR/GritLM-style: callable template must be called on empty input
    so it can emit a non-empty document prefix (`<|embed|>\\n` for them).
    """

    def template(instr, prompt_type):
        return f"<|user|>\n{instr}\n<|embed|>\n" if instr else "<|embed|>\n"

    enc = _FakeEncoder(instruction_template=template)
    # task only specifies "query"; document falls back to empty instruction
    meta = _meta(prompt={"query": "Given a biology post, retrieve relevant passages"})
    assert enc.get_task_instruction(meta, PromptType.document) == "<|embed|>\n"
    # query path still works
    assert (
        enc.get_task_instruction(meta, PromptType.query)
        == "<|user|>\nGiven a biology post, retrieve relevant passages\n<|embed|>\n"
    )


def test_str_template_gated_for_empty_instruction():
    """E5/Qwen-instruct-style: `str` templates contain `{instruction}` and
    would produce malformed prefixes like `"Instruct: \\nQuery: "` if
    formatted with empty input. The gate must still drop them.
    """
    enc = _FakeEncoder(instruction_template="Instruct: {instruction}\nQuery: ")
    meta = _meta(prompt={"query": "Some query instruction"})
    # document side has empty instruction → str template stays gated
    assert not enc.get_task_instruction(meta, PromptType.document)
    # query side renders the template
    assert (
        enc.get_task_instruction(meta, PromptType.query)
        == "Instruct: Some query instruction\nQuery: "
    )


def test_no_template_returns_raw_instruction():
    """No instruction_template at all — get_task_instruction returns the raw
    instruction (empty for documents under query-only task prompts).
    """
    enc = _FakeEncoder(instruction_template=None)
    meta = _meta(prompt={"query": "Q"})
    assert not enc.get_task_instruction(meta, PromptType.document)
    assert enc.get_task_instruction(meta, PromptType.query) == "Q"
