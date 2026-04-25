from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class TUNABenchV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TUNABenchV2TRetrieval",
        description=(
            "Retrieve the fine-grained English caption that describes a given video "
            "from the TUNA-Bench dataset of dense dynamic videos with detailed "
            "temporal descriptions."
        ),
        reference="https://arxiv.org/abs/2505.20124",
        dataset={
            "path": "mteb/TUNA-Bench_1K",
            "revision": "0a2b6ec66a3f662f68ac0f0649020b3c37d8551c",
        },
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ye2025tuna,
  author = {Ye, Jinghao and Zhu, Yanbin and Liu, Jiaqi and Zhang, Yixin and Huang, Qianyu and Zhou, Jianfeng},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos},
  year = {2025},
}
""",
        prompt={
            "query": "Find the caption that best describes the following video.",
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the TUNA-Bench dataset for video-to-text retrieval.

        TODO: Reupload dataset in standard format and remove this custom load_data.
        """
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split=self.metadata.eval_splits[0],
        )
        dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

        query = dataset.select_columns(["id", "video"])
        corpus = dataset.select_columns(["id", "caption"]).rename_column(
            "caption", "text"
        )
        qrels = {}
        for i in range(len(dataset)):
            key = str(i)
            if key not in qrels:
                qrels[key] = {}
            qrels[key][key] = 1
        self.dataset["default"]["test"] = RetrievalSplitData(
            queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
        )
        self.data_loaded = True
