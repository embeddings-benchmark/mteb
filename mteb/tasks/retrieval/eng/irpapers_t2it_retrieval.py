from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class IRPapersT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IRPapersT2ITRetrieval",
        description="IRPAPERS is a collection of 166 Information Retrieval papers spanning 3,230 pages. Each page in the dataset is jointly represented as a base64 encoded string of the page image as well as an OCR-derived text transcription. IRPAPERS also contains 180 needle-in-the-haystack queries.",
        reference="https://arxiv.org/pdf/2602.17687",
        dataset={
            "path": "mteb/IRPapersRetrieval",
            "revision": "fc9ef6144f0df1b44782d6bd5b251f526dea8d54",
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

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        self.dataset["default"]["train"]["corpus"] = self.dataset["default"]["train"][
            "corpus"
        ].rename_column("transcription", "text")

        self.dataset["default"]["train"]["corpus"] = self.dataset["default"]["train"][
            "corpus"
        ].filter(
            lambda x: x["text"] is not None,
            num_proc=num_proc,
        )
