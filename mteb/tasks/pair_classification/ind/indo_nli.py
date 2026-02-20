from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class IndoNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="indonli",
        dataset={
            "path": "mteb/indonli",
            "revision": "cb76e5ca05b56d4f1e0ecaee4d03c1167f162ea6",
        },
        description="IndoNLI is the first human-elicited Natural Language Inference (NLI) dataset for Indonesian. IndoNLI is annotated by both crowd workers and experts.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test_expert"],
        eval_langs=["ind-Latn"],
        main_score="max_ap",
        date=("2021-01-01", "2021-11-01"),  # best guess
        domains=["Encyclopaedic", "Web", "News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{mahendra-etal-2021-indonli,
  address = {Online and Punta Cana, Dominican Republic},
  author = {Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  month = nov,
  pages = {10511--10527},
  publisher = {Association for Computational Linguistics},
  title = {{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian},
  url = {https://aclanthology.org/2021.emnlp-main.821},
  year = {2021},
}
""",
    )
