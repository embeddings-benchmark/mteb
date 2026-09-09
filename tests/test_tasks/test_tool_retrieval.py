"""Guards on the ToolRet setup that determines whether published numbers reproduce.

ToolRet (arXiv:2503.01763) scores a model by retrieving from the *pooled* tool
corpus and taking the unweighted mean over its 35 retrieval tasks. Both choices
matter: scoring against per-category corpora, or micro-averaging over queries,
inflates NDCG@10 by roughly 10 points. Modelling the 35 tasks as MTEB subsets
over one shared corpus is what makes MTEB's cross-subset mean match the paper.

Reference NDCG@10 for `sentence-transformers/all-MiniLM-L6-v2`, macro-averaged
within each category (paper value in parentheses):

    ToolRetrieval             web 14.18 (11.66)  code 14.72 (14.44)  cust. 21.60 (22.80)
    ToolRetrievalInstruction  web 18.29 (12.77)  code 31.88 (31.59)  cust. 31.83 (32.24)

The remaining `web` gap is `max_seq_length`, not the task: this model's card sets
256, while the reference implementation uses ``min(max_position_embeddings, 2048)``
= 512, and `web` has by far the longest tool documents. Re-running
ToolRetrievalInstruction with ``model.model.max_seq_length = 512`` gives
web 13.23, code 32.38, cust. 33.29 -- mean |delta| 0.77 against Table 5.
"""

import pytest

import mteb
from mteb.tasks.retrieval.eng.tool_retrieval import (
    _TASK_2_CATEGORY,
    _TOOL_CATEGORIES,
)

TASK_NAMES = ["ToolRetrieval", "ToolRetrievalInstruction"]
POOLED_CORPUS_SIZE = 44453

# Query counts in the released dataset. 34 of 35 agree with the paper's Table 6;
# `autotools-music` ships 32 queries where Table 6 reports 72.
RELEASED_QUERY_COUNTS = {
    "apibank": 101, "apigen": 1000, "appbench": 32, "autotools-food": 22,
    "autotools-music": 32, "autotools-weather": 11, "craft-math-algebra": 280,
    "craft-tabmwp": 174, "craft-vqa": 200, "gorilla-huggingface": 500,
    "gorilla-pytorch": 43, "gorilla-tensor": 55, "gpt4tools": 32, "gta": 14,
    "metatool": 200, "mnms": 33, "restgpt-spotify": 40, "restgpt-tmdb": 54,
    "reversechain": 200, "rotbench": 550, "t-eval-dialog": 50, "t-eval-step": 50,
    "taskbench-daily": 40, "taskbench-huggingface": 23, "taskbench-multimedia": 40,
    "tool-be-honest": 350, "toolace": 1000, "toolalpaca": 94, "toolbench": 1100,
    "toolbench-sam": 197, "toolemu": 38, "tooleyes": 95, "toolink": 497,
    "toollens": 314, "ultratool": 500,
}


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_subsets_are_the_retrieval_tasks(task_name: str) -> None:
    """Subsets must be the 35 tasks, not the 3 categories.

    MTEB averages across subsets, so this is what reproduces the paper's
    unweighted mean. Collapsing to categories would micro-average instead.
    """
    task = mteb.get_tasks(tasks=[task_name])[0]
    assert set(task.metadata.eval_langs) == set(_TASK_2_CATEGORY)
    assert len(task.metadata.eval_langs) == 35


def test_categories_are_only_a_reporting_grouping() -> None:
    counts = {c: 0 for c in _TOOL_CATEGORIES}
    for category in _TASK_2_CATEGORY.values():
        counts[category] += 1
    assert counts == {"web": 19, "code": 7, "customized": 9}


@pytest.mark.test_datasets
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_every_subset_retrieves_from_the_pooled_corpus(task_name: str) -> None:
    """All 35 subsets must share one corpus covering every category's tools."""
    task = mteb.get_tasks(tasks=[task_name])[0]
    task.load_data()

    corpora = {
        subset: task.dataset[subset]["test"]["corpus"] for subset in _TASK_2_CATEGORY
    }
    for subset, corpus in corpora.items():
        assert len(corpus) == POOLED_CORPUS_SIZE, (
            f"{subset} retrieves from {len(corpus)} tools, expected the pooled "
            f"{POOLED_CORPUS_SIZE}; per-category corpora inflate NDCG@10 by ~10 points"
        )


@pytest.mark.test_datasets
def test_query_counts_match_the_released_dataset() -> None:
    task = mteb.get_tasks(tasks=["ToolRetrieval"])[0]
    task.load_data()
    actual = {
        subset: len(task.dataset[subset]["test"]["queries"])
        for subset in _TASK_2_CATEGORY
    }
    assert actual == RELEASED_QUERY_COUNTS
    assert sum(actual.values()) == 7961


@pytest.mark.test_datasets
def test_relevance_judgements_resolve_into_the_corpus() -> None:
    task = mteb.get_tasks(tasks=["ToolRetrieval"])[0]
    task.load_data()
    for subset in _TASK_2_CATEGORY:
        split = task.dataset[subset]["test"]
        corpus_ids = set(split["corpus"]["id"])
        judged = {doc_id for rels in split["relevant_docs"].values() for doc_id in rels}
        assert judged <= corpus_ids, f"{subset} has qrels outside the corpus"


@pytest.mark.test_datasets
def test_instruction_variant_uses_the_reference_prompt_template() -> None:
    """The `w/ inst.` task bakes the paper's template into the query text.

    MTEB's default instruction handling would join query and instruction as
    ``"{query} {instruction}"``, which scores ~2.4 NDCG@10 below the published
    numbers on the code and customized subsets. Emitting an ``instruction``
    column instead of prebuilt text would silently reintroduce that gap.
    """
    plain = mteb.get_tasks(tasks=["ToolRetrieval"])[0]
    plain.load_data()
    with_inst = mteb.get_tasks(tasks=["ToolRetrievalInstruction"])[0]
    with_inst.load_data()

    subset = next(iter(_TASK_2_CATEGORY))
    plain_queries = plain.dataset[subset]["test"]["queries"]
    inst_queries = with_inst.dataset[subset]["test"]["queries"]

    # neither variant exposes a separate column; the template is applied by the task
    assert "instruction" not in plain_queries.column_names
    assert "instruction" not in inst_queries.column_names

    assert all(
        text.startswith("Instruct: ") and "\nQuery: " in text
        for text in inst_queries["text"]
    )
    # the plain variant must stay untemplated
    assert not any(text.startswith("Instruct: ") for text in plain_queries["text"])
    # and the instruction must add information beyond the bare query
    assert all(
        len(inst) > len(bare)
        for inst, bare in zip(inst_queries["text"], plain_queries["text"])
    )
