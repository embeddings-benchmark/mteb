from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class PerceptionTestVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PerceptionTestVideoCentricQA",
        description="Perception Test is a multimodal benchmark designed to evaluate perception and reasoning skills of video models across memory, abstraction, physics, and semantics. Each example pairs a video with a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2305.13786",
        dataset={
            "path": "mteb/PerceptionTest_val",
            "revision": "547b8ec008265feb6957a279dd9ab1403c956998",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-05-23", "2023-05-23"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{patraucean2024perception,
  author = {Patraucean, Viorica and Smaira, Lucas and Gupta, Ankush and Recasens, Adria and Markeeva, Larisa and Banarse, Dylan and Koppula, Skanda and Malinowski, Mateusz and Yang, Yi and Doersch, Carl and others},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Perception Test: A Diagnostic Benchmark for Multimodal Video Models},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            ds = load_dataset(
                self.metadata.dataset["path"],
                revision=self.metadata.dataset["revision"],
                split=split,
            )
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


class PerceptionTestVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PerceptionTestVideoAudioCentricQA",
        description="Perception Test is a multimodal benchmark designed to evaluate perception and reasoning skills of video models across memory, abstraction, physics, and semantics. Each example pairs a video with audio and a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2305.13786",
        dataset={
            "path": "mteb/PerceptionTest_val",
            "revision": "547b8ec008265feb6957a279dd9ab1403c956998",
        },
        type="VideoAudioCentricQA",
        category="atv2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-05-23", "2023-05-23"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{patraucean2024perception,
  author = {Patraucean, Viorica and Smaira, Lucas and Gupta, Ankush and Recasens, Adria and Markeeva, Larisa and Banarse, Dylan and Koppula, Skanda and Malinowski, Mateusz and Yang, Yi and Doersch, Carl and others},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Perception Test: A Diagnostic Benchmark for Multimodal Video Models},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            ds = load_dataset(
                self.metadata.dataset["path"],
                revision=self.metadata.dataset["revision"],
                split=split,
            )
            ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

            queries = ds.select_columns(["id", "question", "video", "audio"]).rename_column(
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
