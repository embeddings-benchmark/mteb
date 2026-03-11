import base64

from datasets import Image, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
):
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        corpus_ds = load_dataset(
            path,
            "docs",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{x['pdf_id_x']}-{x['page_number']}",
                "text": x["transcription"] or "",
                "image": {"bytes": base64.b64decode(x["base64_str"])},
                "modality": "image, text",
            },
            remove_columns=[
                "dataset_id",
                "pdf_id_x",
                "pdf_name",
                "pdf_title",
                "page_number",
                "base64_str",
                "base64_bytes",
                "transcription",
                "transcription_input_tokens",
                "transcription_output_tokens",
                "pdf_id_y",
                "error",
            ],
        )
        corpus_ds = corpus_ds.cast_column("image", Image())
        corpus[split] = corpus_ds

        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )

        relevant_docs[split] = {}
        for idx, row in enumerate(query_ds):
            qid = f"query-{idx}"
            did = f"corpus-{row['pdf_id']}-{row['page_number']}"
            relevant_docs[split][qid] = {did: 1}

        query_ds = query_ds.map(
            lambda x, idx: {
                "id": f"query-{idx}",
                "text": x["question"],
                "modality": "text",
            },
            with_indices=True,
            remove_columns=[
                "dataset_id",
                "pdf_id",
                "pdf_title",
                "page_number",
                "question",
                "answer",
            ],
        )
        queries[split] = query_ds

    return corpus, queries, relevant_docs


class IRPapersRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IRPapersRetrieval",
        description="IRPAPERS is a collection of 166 Information Retrieval papers spanning 3,230 pages. Each page in the dataset is jointly represented as a base64 encoded string of the page image as well as an OCR-derived text transcription. IRPAPERS also contains 180 needle-in-the-haystack queries.",
        reference="https://arxiv.org/pdf/2602.17687",
        dataset={
            "path": "weaviate/IRPAPERS",
            "revision": "7d8ca2f6dd9efded3e27013d15782d584f93e9da",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=("2026-01-01", "2026-01-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{shorten2026,
  archiveprefix = {arXiv},
  author = {Connor Shorten and Augustas Skaburskas and Daniel M. Jones and Charles Pierse and Roberto Esposito and John Trengrove and Etienne Dilocker and Bob van Luijt},
  eprint = {2602.17687},
  primaryclass = {cs.IR},
  title = {IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and Question Answering},
  url = {https://arxiv.org/pdf/2602.17687},
  year = {2026},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )

        self.data_loaded = True
