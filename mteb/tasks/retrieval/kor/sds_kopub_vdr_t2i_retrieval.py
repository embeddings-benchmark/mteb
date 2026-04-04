from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    eval_splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
):
    split = eval_splits[0]

    qa_ds = load_dataset(
        path,
        name="SDS-KoPub-QA",
        split=split,
        revision=revision,
        num_proc=num_proc,
    )
    corpus_ds = load_dataset(
        path,
        name="SDS-KoPub-corpus",
        split=split,
        revision=revision,
        num_proc=num_proc,
    )

    corpus_ds = corpus_ds.select_columns(["id", "image"])
    corpus_ds = corpus_ds.map(lambda x: {"modality": "image"}, num_proc=num_proc)

    qa_ds = qa_ds.map(
        lambda example, idx: {
            "id": f"query-{idx}",
            "text": example["query"],
            "corpus_id": example["id"],
        },
        with_indices=True,
        num_proc=num_proc,
    )

    return (
        {split: corpus_ds},
        {split: qa_ds.select_columns(["id", "text"])},
        {split: {row["id"]: {row["corpus_id"]: 1} for row in qa_ds}},
    )


class SDSKoPubVDRT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SDSKoPubVDRT2IRetrieval",
        description="SDS KoPub-VDR is a benchmark dataset for Visual Document Retrieval (VDR) in the context of Korean public documents. It contains real-world government document images paired with natural-language queries, corresponding answer pages, and ground-truth answers. The dataset is designed to evaluate AI models that go beyond simple text matching, requiring comprehensive understanding of visual layouts, tables, graphs, and images to accurately locate relevant information.",
        reference="https://arxiv.org/abs/2511.04910",
        dataset={
            "path": "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
            "revision": "759fcae092aef58436d125e72f74a2b53cdd5640",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-11-07", "2025-11-11"),
        domains=["Government", "Legal", "Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{lee2025sdskopubvdrbenchmark,
  archiveprefix = {arXiv},
  author = {Jaehoon Lee and Sohyun Kim and Wanggeun Park and Geon Lee and Seungkyung Kim and Minyoung Lee},
  eprint = {2511.04910},
  primaryclass = {cs.CL},
  title = {SDS KoPub VDR: A Benchmark Dataset for Visual Document Retrieval in Korean Public Documents},
  url = {https://arxiv.org/abs/2511.04910},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.data_loaded = True
