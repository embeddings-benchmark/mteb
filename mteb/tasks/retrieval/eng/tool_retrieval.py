from __future__ import annotations

import json
from typing import Any

import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_QUERIES_DATASET = "mangopy/ToolRet-Queries"
_QUERIES_REVISION = "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445"
_TOOLS_DATASET = "mangopy/ToolRet-Tools"
_TOOLS_REVISION = "e06c38c75612b6536bd959e08cdd345894aba6a7"

_TASK_2_CATEGORY: dict[str, str] = {
    "craft-math-algebra": "code",
    "craft-tabmwp": "code",
    "craft-vqa": "code",
    "gorilla-huggingface": "code",
    "gorilla-pytorch": "code",
    "gorilla-tensor": "code",
    "toolink": "code",
    "appbench": "customized",
    "gpt4tools": "customized",
    "gta": "customized",
    "taskbench-huggingface": "customized",
    "taskbench-multimedia": "customized",
    "metatool": "customized",
    "tool-be-honest": "customized",
    "toolalpaca": "customized",
    "toolbench-sam": "customized",
    "apibank": "web",
    "apigen": "web",
    "mnms": "web",
    "reversechain": "web",
    "rotbench": "web",
    "t-eval-dialog": "web",
    "t-eval-step": "web",
    "taskbench-daily": "web",
    "toolace": "web",
    "toolbench": "web",
    "toolemu": "web",
    "tooleyes": "web",
    "toollens": "web",
    "ultratool": "web",
    "autotools-food": "web",
    "autotools-music": "web",
    "autotools-weather": "web",
    "restgpt-spotify": "web",
    "restgpt-tmdb": "web",
}

_EVAL_LANGS: dict[str, list[str]] = {
    "code": ["eng-Latn"],
    "customized": ["eng-Latn"],
    "web": ["eng-Latn"],
}


class ToolRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ToolRetrieval",
        description="Tool Retrieval Benchmark for Large Language Models (ToolRet), evaluating retrieval models on identifying appropriate tools for LLMs across code, customized, and web domains.",
        reference="https://arxiv.org/abs/2503.01763",
        dataset={
            "path": _QUERIES_DATASET,
            "revision": _QUERIES_REVISION,
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
        bibtex_citation=r"""@article{shi2025toolret,
  author = {Shi, Zhengliang and Wang, Yuhan and Yan, Lingyong and Ren, Pengjie and Wang, Shuaiqiang and Yin, Dawei and Ren, Zhaochun},
  journal = {arXiv preprint arXiv:2503.01763},
  title = {Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  year = {2025}
}""",
        prompt={
            "query": "Given a user query, retrieve relevant tool documentation that can solve the query"
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset: dict[str, dict[str, RetrievalSplitData]] = {}

        for category in self.hf_subsets:
            tools_ds = datasets.load_dataset(
                _TOOLS_DATASET,
                category,
                split="tools",
                revision=_TOOLS_REVISION,
            )
            corpus = Dataset.from_dict(
                {
                    "id": [str(doc_id) for doc_id in tools_ds["id"]],
                    "text": tools_ds["documentation"],
                    "title": [""] * len(tools_ds),
                }
            )

            tasks = [
                task_name
                for task_name, cat in _TASK_2_CATEGORY.items()
                if cat == category
            ]

            q_ids: list[str] = []
            q_texts: list[str] = []
            relevant_docs: dict[str, dict[str, int]] = {}

            for task_name in tasks:
                queries_ds = datasets.load_dataset(
                    _QUERIES_DATASET,
                    task_name,
                    split="queries",
                    revision=_QUERIES_REVISION,
                )
                for row in queries_ds:
                    qid = str(row["id"])
                    q_ids.append(qid)
                    q_texts.append(row["query"])
                    labels = json.loads(row["labels"])
                    relevant_docs[qid] = {
                        str(item["id"]): int(item["relevance"]) for item in labels
                    }

            queries = Dataset.from_dict(
                {
                    "id": q_ids,
                    "text": q_texts,
                }
            )

            self.dataset[category] = {
                "test": RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }

        self.data_loaded = True
