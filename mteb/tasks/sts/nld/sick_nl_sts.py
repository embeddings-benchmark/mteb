from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class SICKNLSTS(AbsTaskSTS):
    fast_loading = True
    metadata = TaskMetadata(
        name="SICK-NL-STS",
        dataset={
            "path": "clips/mteb-nl-sick-sts-pr",
            "revision": "7f88f003fc4e37ed8cd9ade84e390d871b032fef",
        },
        description="SICK-NL (read: signal), a dataset targeting Natural Language Inference in Dutch. SICK-NL is "
        "obtained by translating the SICK dataset of (Marelli et al., 2014) from English into Dutch.",
        reference="https://aclanthology.org/2021.eacl-main.126/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2017-12-31"),
        domains=["News", "Social", "Web", "Spoken", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{wijnholds2021sick,
  author = {Wijnholds, Gijs and Moortgat, Michael},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages = {1474--1479},
  title = {SICK-NL: A Dataset for Dutch Natural Language Inference},
  year = {2021},
}
""",
        prompt={"query": "Haal semantisch vergelijkbare tekst op"},
    )

    min_score = 0
    max_score = 5
