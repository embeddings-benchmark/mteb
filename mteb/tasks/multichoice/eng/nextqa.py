from __future__ import annotations

from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
):
    corpus: dict[str, Dataset] = {}
    queries: dict[str, Dataset] = {}
    relevant_docs: dict[str, dict[str, dict[str, int]]] = {}
    top_ranked: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for split in splits:
        ds = load_dataset(path, revision=revision, split=split)
        ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

        queries[split] = ds.select_columns(["id", "question", "video"]).rename_column(
            "question", "text"
        )

        text_view = ds.select_columns(["id", "candidates", "answer"])
        corpus_rows: list[dict] = []
        relevant_docs[split] = {}
        for row in text_view:
            qid = row["id"]
            answer = row["answer"]
            relevant_docs[split][qid] = {}
            for j, candidate in enumerate(row["candidates"]):
                doc_id = f"{qid}_c{j}"
                corpus_rows.append({"id": doc_id, "text": candidate})
                top_ranked[split][qid].append(doc_id)
                relevant_docs[split][qid][doc_id] = 1 if candidate == answer else 0

        corpus[split] = Dataset.from_list(corpus_rows)

    top_ranked_plain = {s: dict(d) for s, d in top_ranked.items()}
    return corpus, queries, relevant_docs, top_ranked_plain


class NExTQAVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NExTQAVideoCentricQA",
        description="NExT-QA is a video question answering benchmark targeting causal and temporal reasoning over everyday videos. Each example pairs a short video with a natural language question and 5 candidate answers, of which exactly one is correct. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate from its 5 choices.",
        reference="https://arxiv.org/abs/2105.08276",
        dataset={
            "path": "mteb/NExT-QA",
            "revision": "18efe467c7dfd207d3e1cd6642d5ba3b31e8b25d",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-05-18", "2021-05-18"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{xiao2021next,
  author = {Xiao, Junbin and Shang, Xindi and Yao, Angela and Chua, Tat-Seng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {9777-9786},
  title = {NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
  year = {2021},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
