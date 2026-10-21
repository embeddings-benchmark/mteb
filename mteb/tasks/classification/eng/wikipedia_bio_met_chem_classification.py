from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class WikipediaBioMetChemClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaBioMetChemClassification",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html",
        dataset={
            "path": "BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification",
            "revision": "6ac491e5de9070c6dd434b31e76d3d379123dcff",
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
        superseded_by="WikipediaBioMetChemClassification.v2",
    )


class WikipediaBioMetChemClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaBioMetChemClassification.v2",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://proceedings.mlr.press/v262/shiraee-kasmaee24a.html",
        dataset={
            "path": "mteb/wikipedia_bio_met_chem",
            "revision": "5341bdf799e94949b5c2684bb54bfe462c528179",
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
        adapted_from=["WikipediaBioMetChemClassification"],
    )
