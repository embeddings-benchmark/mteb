from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class WorldQAVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WorldQAVideoCentricQA",
        description="WorldQA is a multimodal long video benchmark evaluating world knowledge through long-chain reasoning. Each example pairs a video with a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2405.03272",
        dataset={
            "path": "mteb/worldqa",
            "revision": "2c8dbaaec94aae36e55d45c8440c7560a8b077ce",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-05-06", "2024-05-06"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{zhang2024worldqa,
  author = {Zhang, Yuanhan and Zhang, Kaichen and Li, Bo and Pu, Fanyi and Setiadharma, Christopher Arif and Yang, Jingkang and Liu, Ziwei},
  journal = {arXiv preprint arXiv:2405.03272},
  title = {WorldQA: Multimodal World Knowledge in Videos through Long-Chain Reasoning},
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


class WorldQAVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WorldQAVideoAudioCentricQA",
        description="WorldQA is a multimodal long video benchmark evaluating world knowledge through long-chain reasoning. Each example pairs a video with audio and a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2405.03272",
        dataset={
            "path": "mteb/worldqa",
            "revision": "2c8dbaaec94aae36e55d45c8440c7560a8b077ce",
        },
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-05-06", "2024-05-06"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{zhang2024worldqa,
  author = {Zhang, Yuanhan and Zhang, Kaichen and Li, Bo and Pu, Fanyi and Setiadharma, Christopher Arif and Yang, Jingkang and Liu, Ziwei},
  journal = {arXiv preprint arXiv:2405.03272},
  title = {WorldQA: Multimodal World Knowledge in Videos through Long-Chain Reasoning},
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
