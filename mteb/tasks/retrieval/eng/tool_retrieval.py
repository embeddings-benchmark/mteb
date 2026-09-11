from __future__ import annotations

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "craft-math-algebra": ["eng-Latn"],
    "craft-tabmwp": ["eng-Latn"],
    "craft-vqa": ["eng-Latn"],
    "gorilla-huggingface": ["eng-Latn"],
    "gorilla-pytorch": ["eng-Latn"],
    "gorilla-tensor": ["eng-Latn"],
    "toolink": ["eng-Latn"],
    "appbench": ["eng-Latn"],
    "gpt4tools": ["eng-Latn"],
    "gta": ["eng-Latn"],
    "taskbench-huggingface": ["eng-Latn"],
    "taskbench-multimedia": ["eng-Latn"],
    "metatool": ["eng-Latn"],
    "tool-be-honest": ["eng-Latn"],
    "toolalpaca": ["eng-Latn"],
    "toolbench-sam": ["eng-Latn"],
    "apibank": ["eng-Latn"],
    "apigen": ["eng-Latn"],
    "mnms": ["eng-Latn"],
    "reversechain": ["eng-Latn"],
    "rotbench": ["eng-Latn"],
    "t-eval-dialog": ["eng-Latn"],
    "t-eval-step": ["eng-Latn"],
    "taskbench-daily": ["eng-Latn"],
    "toolace": ["eng-Latn"],
    "toolbench": ["eng-Latn"],
    "toolemu": ["eng-Latn"],
    "tooleyes": ["eng-Latn"],
    "toollens": ["eng-Latn"],
    "ultratool": ["eng-Latn"],
    "autotools-food": ["eng-Latn"],
    "autotools-music": ["eng-Latn"],
    "autotools-weather": ["eng-Latn"],
    "restgpt-spotify": ["eng-Latn"],
    "restgpt-tmdb": ["eng-Latn"],
}


class ToolRetrieval(AbsTaskRetrieval):
    """ToolRet, `w/o inst.` setting (paper Table 4)."""

    metadata = TaskMetadata(
        name="ToolRetrieval",
        description="Tool Retrieval Benchmark for Large Language Models (ToolRet), evaluating retrieval models on identifying appropriate tools for LLMs across code, customized, and web domains.",
        reference="https://arxiv.org/abs/2503.01763",
        dataset={
            "path": "mteb/ToolRetrieval",
            "revision": "76d45e560059754e289ea202462865a585679619",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-03-03"),
        domains=["Programming", "Web", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{shi2025toolret,
  author = {Shi, Zhengliang and Wang, Yuhan and Yan, Lingyong and Ren, Pengjie and Wang, Shuaiqiang and Yin, Dawei and Ren, Zhaochun},
  journal = {arXiv preprint arXiv:2503.01763},
  title = {Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  year = {2025},
}
""",
        prompt={
            "query": "Given a user query, retrieve relevant tool documentation that can solve the query"
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset:
            for split in self.dataset[subset]:
                self.dataset[subset][split]["queries"] = self.dataset[subset][split][
                    "queries"
                ].remove_columns(["instruction"])


class ToolRetrievalInstruction(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ToolRetrievalInstruction",
        description="ToolRet in the `w/ inst.` setting, where each query is paired with a task instruction describing what kind of tool is required. Corresponds to Table 5 of the paper; see ToolRetrieval for the `w/o inst.` setting.",
        reference="https://arxiv.org/abs/2503.01763",
        dataset={
            "path": "mteb/ToolRetrieval",
            "revision": "76d45e560059754e289ea202462865a585679619",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-03-03"),
        domains=["Programming", "Web", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{shi2025toolret,
  author = {Shi, Zhengliang and Wang, Yuhan and Yan, Lingyong and Ren, Pengjie and Wang, Shuaiqiang and Yin, Dawei and Ren, Zhaochun},
  journal = {arXiv preprint arXiv:2503.01763},
  title = {Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  year = {2025},
}
""",
        prompt={
            "query": "Given a user query, retrieve relevant tool documentation that can solve the query"
        },
    )
