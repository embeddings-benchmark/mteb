from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TRECCOVIDNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-NL",
        description="TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific "
        "articles related to the COVID-19 pandemic. TRECCOVID-NL is a Dutch translation. ",
        reference="https://colab.research.google.com/drive/1R99rjeAGt8S9IfAIRR3wS052sNu3Bjo-#scrollTo=4HduGW6xHnrZ",
        dataset={
            "path": "clips/beir-nl-trec-covid",
            "revision": "04dd804c048866b0ab90a55ded77789485828281",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2019-12-01",
            "2022-12-31",
        ),  # approximate date of covid pandemic start and end (best guess)
        domains=["Medical", "Academic", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@inproceedings{banar2024beirnlzeroshotinformationretrieval,
  address = {Abu Dhabi, UAE},
  archiveprefix = {arXiv},
  author = {Lotfi, Ehsan and Banar, Nikolay and Daelemans, Walter},
  booktitle = {Proceedings of the 18th Workshop on Building and Using Comparable Corpora ({BUCC})},
  editor = {Sharoff, Serge and Terryn, Ayla Rigouts and Zweigenbaum, Pierre and Rapp, Reinhard},
  eprint = {2412.08329},
  month = {jan},
  pages = {36--45},
  publisher = {Association for Computational Linguistics},
  title = {{BEIR}-{NL}: Zero-shot Information Retrieval Benchmark for the {D}utch Language},
  url = {https://aclanthology.org/2025.bucc-1.5/},
  year = {2025},
}
""",
        adapted_from=["TRECCOVID"],
    )
