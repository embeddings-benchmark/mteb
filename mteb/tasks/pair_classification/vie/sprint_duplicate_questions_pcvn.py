from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SprintDuplicateQuestionsPCVN(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SprintDuplicateQuestions-VN",
        description="A translated dataset from Duplicate questions from the Sprint community. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "GreenNode/sprintduplicatequestions-pairclassification-vn",
            "revision": "2552beae0e4fe7fe05d088814f78a4c309ad2219",
        },
        type="PairClassification",
        category="t2c",
        eval_splits=["validation", "test"],
        eval_langs=["vie-Latn"],
        main_score="max_ap",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Programming", "Written"],
        task_subtypes=["Duplicate Detection"],
        bibtex_citation=r"""
@inproceedings{pham2025vnmtebvietnamesemassivetext,
  address = {Rabat, Morocco},
  author = {Pham, Loc and Luu, Tung and Vo, Thu and Nguyen, Minh and Hoang, Viet},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026},
  doi = {10.18653/v1/2026.findings-eacl.86},
  editor = {Demberg, Vera and Inui, Kentaro and Marquez, Llu{\'i}s},
  isbn = {979-8-89176-386-9},
  month = {mar},
  pages = {1705--1725},
  publisher = {Association for Computational Linguistics},
  title = {{VN}-{MTEB}: {V}ietnamese Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2026.findings-eacl.86/},
  year = {2026},
}
""",
        adapted_from=["SprintDuplicateQuestions"],
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
