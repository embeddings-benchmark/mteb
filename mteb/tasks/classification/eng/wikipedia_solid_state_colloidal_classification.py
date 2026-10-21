from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class WikipediaSolidStateColloidalClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaSolidStateColloidalClassification",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html",
        dataset={
            "path": "BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification",
            "revision": "7d8df44e588b6143d4856c781f72f919fa0599a7",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{kasmaee2024chemteb,
  archiveprefix = {arXiv},
  author = {Ali Shiraee Kasmaee and Mohammad Khodadad and Mohammad Arshi Saloot and Nick Sherck and Stephen Dokas and Hamidreza Mahyar and Soheila Samiee},
  booktitle = {ENLSP},
  eprint = {2412.00532},
  pages = {512-531},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance & Efficiency on a Specific Domain},
  url = {https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html},
  year = {2024},
}
""",
    )
