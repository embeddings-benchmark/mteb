from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class EgoSchemaVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EgoSchemaVideoCentricQA",
        description="EgoSchema is a long-form video language understanding benchmark built from egocentric video. Each example pairs a ~3-minute first-person video with a question and 5 candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate from its 5 choices.",
        reference="https://arxiv.org/abs/2308.09126",
        dataset={
            "path": "mteb/EgoSchema_subset",
            "revision": "d4ca5f83a6f065471e090a9c6542894c66d4b7d9",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-17", "2023-08-17"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{mangalam2023egoschema,
  author = {Mangalam, Karttikeya and Akshulakov, Raiymbek and Malik, Jitendra},
  journal = {Advances in Neural Information Processing Systems},
  title = {EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding},
  year = {2023},
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
