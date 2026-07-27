from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from datasets import Dataset, Video, load_dataset
from huggingface_hub import snapshot_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MMVUVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMVUVideoCentricQA",
        description="MMVU is an expert-level, multi-discipline video understanding benchmark with questions spanning 27 subjects across Science, Healthcare, Humanities & Social Sciences, and Engineering. Each multiple-choice example pairs a specialized-domain video with a question and 5 candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate. Used the public validation multiple-choice subset (~625 examples).",
        reference="https://arxiv.org/abs/2501.12380",
        dataset={
            "path": "yale-nlp/MMVU",
            "revision": "b937f414a87e9012acba49d95669020b24fa9ee9",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-21", "2025-01-21"),
        domains=["Academic", "Medical", "Engineering", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@misc{zhao2025mmvu,
  author = {Zhao, Yilun and Xie, Lujing and Zhang, Haowei and Gan, Guo and Long, Yitao and Hu, Zhiyuan and Hu, Tongyan and Chen, Weiyuan and Li, Chuhan and Song, Junyang and Xu, Zhijian and Wang, Chengye and Pan, Weifeng and Shangguan, Ziyao and Tang, Xiangru and Liang, Zhenwen and Liu, Yixin and Zhao, Chen and Cohan, Arman},
  eprint = {2501.12380},
  primaryClass = {cs.CV},
  title = {MMVU: Measuring Expert-Level Multi-Discipline Video Understanding},
  url = {https://arxiv.org/abs/2501.12380},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        def _video_relpath(url: str) -> str:
            path = unquote(urlparse(url).path)
            marker = "/videos/"
            idx = path.lower().rfind(marker)
            if idx < 0:
                raise ValueError(f"Unexpected MMVU video URL: {url}")
            return path[idx + len(marker) :]

        if self.data_loaded:
            return

        ds = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split="validation",
        )
        ds = ds.filter(lambda row: row["question_type"] == "multiple-choice")
        video_rels = sorted({_video_relpath(url) for url in ds["video"]})
        repo_dir = Path(
            snapshot_download(
                repo_id=self.metadata.dataset["path"],
                repo_type="dataset",
                revision=self.metadata.dataset["revision"],
                allow_patterns=[
                    "validation.json",
                    *[f"videos/{rel}" for rel in video_rels],
                ],
            )
        )

        query_rows: list[dict] = []
        corpus_rows: list[dict] = []
        relevant_docs: dict[str, dict[str, int]] = {}
        top_ranked: dict[str, list[str]] = {}

        for i, row in enumerate(ds):
            qid = f"q{i}"
            video_path = repo_dir / "videos" / _video_relpath(row["video"])
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing MMVU video: {video_path}")

            query_rows.append(
                {
                    "id": qid,
                    "text": row["question"],
                    "video": str(video_path),
                }
            )

            answer_text = row["choices"][row["answer"]]
            top_ranked[qid] = []
            choice_keys = ("A", "B", "C", "D", "E")
            for j, key in enumerate(choice_keys):
                doc_id = f"{qid}_c{j}"
                candidate = row["choices"][key]
                corpus_rows.append({"id": doc_id, "text": candidate})
                top_ranked[qid].append(doc_id)
                if candidate == answer_text:
                    relevant_docs[qid] = {doc_id: 1}

        queries = Dataset.from_list(query_rows).cast_column("video", Video())
        corpus = Dataset.from_list(corpus_rows)
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=top_ranked,
                )
            }
        }
        self.data_loaded = True
