from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class AVQAVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVQAVideoCentricQA",
        description="AVQA is an audio-visual question answering benchmark requiring models to understand both the audio and visual content of videos to answer questions. Each example pairs a video with a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2207.10489",
        dataset={
            "path": "mteb/AVQA_val",
            "revision": "91c54bcc1debe3a468ff705ce98ac5e34712fd77",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-07-21", "2022-07-21"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{yang2022avqa,
  author = {Yang, Pinci and Wang, Xin and Duan, Xuguang and Chen, Hong and Hou, Runze and Jin, Cong and Zhu, Wenwu},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  title = {AVQA: A Dataset for Audio-Visual Question Answering on Videos},
  year = {2022},
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
