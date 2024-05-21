from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class JSICK(AbsTaskSTS):
    metadata = TaskMetadata(
        name="JSICK",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "jsick",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
        },
        description="JSICK is the Japanese NLI and STS dataset by manually translating the English dataset SICK (Marelli et al., 2014) into Japanese.",
        reference="https://github.com/sbintuitions/JMTEB",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="cosine_spearman",
        date=("2000-01-01", "2012-12-31"),  # best guess
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @article{yanaka2022compositional,
            title={Compositional Evaluation on Japanese Textual Entailment and Similarity},
            author={Yanaka, Hitomi and Mineshima, Koji},
            journal={Transactions of the Association for Computational Linguistics},
            volume={10},
            pages={1266--1284},
            year={2022},
            publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…}
        }
        """,
        n_samples={"test": 1986},
        avg_character_length={"test": 21.47},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_column("label", "score")
