from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class JSICK(AbsTaskSTS):
    metadata = TaskMetadata(
        name="JSICK",
        dataset={
            "path": "mteb/JSICK",
            "revision": "729cfe4a16d3c2b61c6aa9f9f6c8a96bb5512868",
        },
        description="JSICK is the Japanese NLI and STS dataset by manually translating the English dataset SICK (Marelli et al., 2014) into Japanese.",
        reference="https://github.com/sbintuitions/JMTEB",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="cosine_spearman",
        date=("2000-01-01", "2012-12-31"),  # best guess
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{yanaka2022compositional,
  author = {Yanaka, Hitomi and Mineshima, Koji},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {1266--1284},
  publisher = {MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…},
  title = {Compositional Evaluation on Japanese Textual Entailment and Similarity},
  volume = {10},
  year = {2022},
}
""",
    )

    min_score = 1
    max_score = 5
