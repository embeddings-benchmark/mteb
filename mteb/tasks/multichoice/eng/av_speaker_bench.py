from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class AVSpeakerBenchVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVSpeakerBenchVideoCentricQA",
        description="AV-SpeakerBench is an audio-visual benchmark for speaker-centric question answering in video, evaluating multimodal LLMs on human speech understanding. Each example pairs a video with a question about the speaker and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2512.02231",
        dataset={
            "path": "mteb/AV-SpeakerBench",
            "revision": "46da53344c6a968d30cd72e28862732adf747802",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-12-03", "2024-12-03"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{nguyen2024seehearunderstand,
  author = {Nguyen, Le Thien Phuc and Yu, Zhuoran and Hang, Samuel Low Yu and An, Subin and Lee, Jeongik and Ban, Yohan and Chung, SeungEun and Nguyen, Thanh-Huy and Maeng, JuWan and Lee, Soochahn and Lee, Yong Jae},
  journal = {arXiv preprint arXiv:2512.02231},
  title = {See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models},
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


class AVSpeakerBenchVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVSpeakerBenchVideoAudioCentricQA",
        description="AV-SpeakerBench is an audio-visual benchmark for speaker-centric question answering in video, evaluating multimodal LLMs on human speech understanding. Each example pairs a video with audio and a question about the speaker and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2512.02231",
        dataset={
            "path": "mteb/AV-SpeakerBench",
            "revision": "46da53344c6a968d30cd72e28862732adf747802",
        },
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-12-03", "2024-12-03"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{nguyen2024seehearunderstand,
  author = {Nguyen, Le Thien Phuc and Yu, Zhuoran and Hang, Samuel Low Yu and An, Subin and Lee, Jeongik and Ban, Yohan and Chung, SeungEun and Nguyen, Thanh-Huy and Maeng, JuWan and Lee, Soochahn and Lee, Yong Jae},
  journal = {arXiv preprint arXiv:2512.02231},
  title = {See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models},
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
