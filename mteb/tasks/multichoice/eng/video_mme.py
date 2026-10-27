from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class VideoMMEShortVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VideoMMEShortVideoCentricQA",
        description='Video-MME is a comprehensive evaluation benchmark for multimodal LLMs in video analysis. The short-video subset contains videos under 2 minutes. Each example pairs a video with a question and 4 candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate. Filtered the test split to `duration == "short"` clips, shuffled (seed=42), capped at 2,000 (~900 examples).',
        reference="https://doi.org/10.1109/cvpr52734.2025.02245",
        dataset={
            "path": "mteb/Video-MME_short",
            "revision": "ebbd19d8ea051d64dd075752c274f5f430b3ff7a",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-05-31", "2024-05-31"),
        domains=["Spoken", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{fu2024video,
  author = {Fu, Chaoyou and Dai, Yuhan and Luo, Yongdong and Li, Lei and Ren, Shuhuai and Zhang, Renrui and Wang, Zihan and Zhou, Chenyu and Shen, Yunhang and Zhang, Mengdan and Chen, Peixian and Li, Yanwei and Lin, Shaohui and Zhao, Sirui and Li, Ke and Xu, Tong and Zheng, Xiawu and Chen, Enhong and Shan, Caifeng and He, Ran and Sun, Xing},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  doi = {10.1109/cvpr52734.2025.02245},
  month = {June},
  pages = {24108–24118},
  publisher = {IEEE},
  title = {Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis},
  url = {https://doi.org/10.1109/cvpr52734.2025.02245},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
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


class VideoMMEShortVideoAudioCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VideoMMEShortVideoAudioCentricQA",
        description='Video-MME is a comprehensive evaluation benchmark for multimodal LLMs in video analysis. The short-video subset contains videos under 2 minutes. Each example pairs a video with audio and a question and 4 candidate answers. The task is formulated as multiple-choice retrieval: given the (video, audio, question) tuple, retrieve the correct candidate. Filtered the test split to `duration == "short"` clips, shuffled (seed=42), capped at 2,000 (~900 examples).',
        reference="https://doi.org/10.1109/cvpr52734.2025.02245",
        dataset={
            "path": "mteb/Video-MME_short",
            "revision": "ebbd19d8ea051d64dd075752c274f5f430b3ff7a",
        },
        type="VideoCentricQA",
        category="vat2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-05-31", "2024-05-31"),
        domains=["Spoken", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{fu2024video,
  author = {Fu, Chaoyou and Dai, Yuhan and Luo, Yongdong and Li, Lei and Ren, Shuhuai and Zhang, Renrui and Wang, Zihan and Zhou, Chenyu and Shen, Yunhang and Zhang, Mengdan and Chen, Peixian and Li, Yanwei and Lin, Shaohui and Zhao, Sirui and Li, Ke and Xu, Tong and Zheng, Xiawu and Chen, Enhong and Shan, Caifeng and He, Ran and Sun, Xing},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  doi = {10.1109/cvpr52734.2025.02245},
  month = {June},
  pages = {24108–24118},
  publisher = {IEEE},
  title = {Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis},
  url = {https://doi.org/10.1109/cvpr52734.2025.02245},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
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
