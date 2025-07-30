from __future__ import annotations

from mteb.abstasks.AbsTaskAnySTS import AbsTaskAnySTS
from mteb.abstasks.task_metadata import TaskMetadata


class JSICK(AbsTaskAnySTS):
    metadata = TaskMetadata.model_construct(
        name="JSICK",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "jsick",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
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

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_column("label", "score")
