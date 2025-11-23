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
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["SprintDuplicateQuestions"],
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
