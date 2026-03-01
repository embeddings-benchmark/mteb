from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FarsTail(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FarsTail",
        dataset={
            "path": "mteb/FarsTail",
            "revision": "0fa0863dc160869b5a2d78803b4440ea3c671ff5",
        },
        description="This dataset, named FarsTail, includes 10,367 samples which are provided in both the Persian language as well as the indexed format to be useful for non-Persian researchers. The samples are generated from 3,539 multiple-choice questions with the least amount of annotator interventions in a way similar to the SciTail dataset",
        reference="https://link.springer.com/article/10.1007/s00500-023-08959-3",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2021-01-01", "2021-07-12"),  # best guess
        domains=["Academic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{amirkhani2023farstail,
  author = {Amirkhani, Hossein and AzariJafari, Mohammad and Faridan-Jahromi, Soroush and Kouhkan, Zeinab and Pourjafari, Zohreh and Amirak, Azadeh},
  doi = {10.1007/s00500-023-08959-3},
  journal = {Soft Computing},
  publisher = {Springer},
  title = {FarsTail: a Persian natural language inference dataset},
  year = {2023},
}
""",  # after removing neutral
    )
