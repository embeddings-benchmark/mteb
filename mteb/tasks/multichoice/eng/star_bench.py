from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_CONFIGS = ["feasibility", "interaction", "prediction", "sequence"]


class STARBenchVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchVideoCentricQA",
        description="STAR (Situated Reasoning in Real-World Videos) is a video question answering benchmark covering four question types: Interaction, Sequence, Prediction, and Feasibility. Each example pairs a video with a situated question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset={
            "path": "mteb/star_bench_val",
            "revision": "0f34df8b497e64bcb4385de99de19dd54c36a788",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-01-19", "2023-01-19"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{wu2024star,
  author = {Wu, Bo and Yu, Shoubin and Chen, Zhenfang and Tenenbaum, Joshua B and Gan, Chuang},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {STAR: Situated Reasoning in Real-World Videos},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            splits = []
            for cfg in _CONFIGS:
                splits.append(
                    load_dataset(
                        self.metadata.dataset["path"],
                        cfg,
                        revision=self.metadata.dataset["revision"],
                        split=split,
                    )
                )
            ds = concatenate_datasets(splits)
            ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

            queries = ds.select_columns(["id", "question", "video"]).rename_column(
                "question", "text"
            )

            corpus_rows: list[dict] = []
            relevant_docs: dict[str, dict[str, int]] = {}
            top_ranked: dict[str, list[str]] = {}
            for row in ds.select_columns(["id", "candidates", "answer"]):
                qid = row["id"]
                answer = row["answer"]
                top_ranked[qid] = []
                for j, candidate in enumerate(row["candidates"]):
                    doc_id = f"{qid}_c{j}"
                    corpus_rows.append({"id": doc_id, "text": candidate})
                    top_ranked[qid].append(doc_id)
                    if candidate == answer:
                        relevant_docs[qid] = {doc_id: 1}

            corpus = Dataset.from_list(corpus_rows)
            self.dataset["default"][split] = RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=top_ranked,
            )
        self.data_loaded = True


class STARBenchVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchVideoAudioCentricQA",
        description="STAR (Situated Reasoning in Real-World Videos) is a video question answering benchmark covering four question types: Interaction, Sequence, Prediction, and Feasibility. Each example pairs a video with audio and a situated question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset={
            "path": "mteb/star_bench_val",
            "revision": "0f34df8b497e64bcb4385de99de19dd54c36a788",
        },
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-01-19", "2023-01-19"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{wu2024star,
  author = {Wu, Bo and Yu, Shoubin and Chen, Zhenfang and Tenenbaum, Joshua B and Gan, Chuang},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {STAR: Situated Reasoning in Real-World Videos},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            splits = []
            for cfg in _CONFIGS:
                splits.append(
                    load_dataset(
                        self.metadata.dataset["path"],
                        cfg,
                        revision=self.metadata.dataset["revision"],
                        split=split,
                    )
                )
            ds = concatenate_datasets(splits)
            ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

            queries = ds.select_columns(
                ["id", "question", "video", "audio"]
            ).rename_column("question", "text")

            corpus_rows: list[dict] = []
            relevant_docs: dict[str, dict[str, int]] = {}
            top_ranked: dict[str, list[str]] = {}
            for row in ds.select_columns(["id", "candidates", "answer"]):
                qid = row["id"]
                answer = row["answer"]
                top_ranked[qid] = []
                for j, candidate in enumerate(row["candidates"]):
                    doc_id = f"{qid}_c{j}"
                    corpus_rows.append({"id": doc_id, "text": candidate})
                    top_ranked[qid].append(doc_id)
                    if candidate == answer:
                        relevant_docs[qid] = {doc_id: 1}

            corpus = Dataset.from_list(corpus_rows)
            self.dataset["default"][split] = RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=top_ranked,
            )
        self.data_loaded = True
