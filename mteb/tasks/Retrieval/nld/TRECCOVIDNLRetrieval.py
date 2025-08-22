from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="s2p",
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
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
        adapted_from=["TRECCOVID"],
    )
