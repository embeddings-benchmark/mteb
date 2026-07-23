from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{wu2024star,
  author = {Wu, Bo and Yu, Shoubin and Chen, Zhenfang and Tenenbaum, Joshua B and Gan, Chuang},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {STAR: Situated Reasoning in Real-World Videos},
  year = {2024},
}
"""

_DATASET = {
    "path": "mteb/star_bench_val",
    "revision": "0f34df8b497e64bcb4385de99de19dd54c36a788",
}

_DATE = ("2023-01-19", "2023-01-19")


def _load_split(path, revision, config, split, modalities):
    ds = load_dataset(path, config, revision=revision, split=split)
    ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

    query_cols = ["id", "question", "video"]
    if "audio" in modalities:
        query_cols.append("audio")
    queries = ds.select_columns(query_cols).rename_column("question", "text")

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
    return RetrievalSplitData(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        top_ranked=top_ranked,
    )


class STARBenchFeasibilityVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchFeasibilityVideoCentricQA",
        description="STAR Feasibility subset: questions asking whether an action is feasible given the video context. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "feasibility", split, ["video", "text"]
            )
        self.data_loaded = True


class STARBenchFeasibilityVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchFeasibilityVideoAudioCentricQA",
        description="STAR Feasibility subset: questions asking whether an action is feasible given the video context. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "feasibility", split, ["video", "audio", "text"]
            )
        self.data_loaded = True


class STARBenchInteractionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchInteractionVideoCentricQA",
        description="STAR Interaction subset: questions about human-object interactions observed in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "interaction", split, ["video", "text"]
            )
        self.data_loaded = True


class STARBenchInteractionVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchInteractionVideoAudioCentricQA",
        description="STAR Interaction subset: questions about human-object interactions observed in the video. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "interaction", split, ["video", "audio", "text"]
            )
        self.data_loaded = True


class STARBenchPredictionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchPredictionVideoCentricQA",
        description="STAR Prediction subset: questions requiring prediction of what will happen next given the video context. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "prediction", split, ["video", "text"]
            )
        self.data_loaded = True


class STARBenchPredictionVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchPredictionVideoAudioCentricQA",
        description="STAR Prediction subset: questions requiring prediction of what will happen next given the video context. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "prediction", split, ["video", "audio", "text"]
            )
        self.data_loaded = True


class STARBenchSequenceVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchSequenceVideoCentricQA",
        description="STAR Sequence subset: questions about the temporal order of actions in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "sequence", split, ["video", "text"]
            )
        self.data_loaded = True


class STARBenchSequenceVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARBenchSequenceVideoAudioCentricQA",
        description="STAR Sequence subset: questions about the temporal order of actions in the video. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2301.08059",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split(
                _DATASET["path"], _DATASET["revision"], "sequence", split, ["video", "audio", "text"]
            )
        self.data_loaded = True
