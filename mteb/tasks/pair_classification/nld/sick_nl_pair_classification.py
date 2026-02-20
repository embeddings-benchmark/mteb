from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SICKNLPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICKNLPairClassification",
        dataset={
            "path": "clips/mteb-nl-sick-pcls-pr",
            "revision": "a13a1892bcb4c077dc416d390389223eea5f20f0",
        },
        description="SICK-NL is a Dutch translation of SICK ",
        reference="https://aclanthology.org/2021.eacl-main.126/",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="max_ap",
        date=("2020-09-01", "2021-01-01"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{wijnholds2021sick,
  author = {Wijnholds, Gijs and Moortgat, Michael},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages = {1474--1479},
  title = {SICK-NL: A Dataset for Dutch Natural Language Inference},
  year = {2021},
}
""",
        prompt={
            "query": "Zoek tekst die semantisch vergelijkbaar is met de gegeven tekst."
        },
    )
