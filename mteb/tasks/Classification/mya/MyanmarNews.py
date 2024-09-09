from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MyanmarNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MyanmarNews",
        dataset={
            "path": "ayehninnkhine/myanmar_news",
            "revision": "b899ec06227db3679b0fe3c4188a6b48cc0b65eb",
            "trust_remote_code": True,
        },
        description="The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.",
        reference="https://huggingface.co/datasets/myanmar_news",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["mya-Mymr"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""""
        @inproceedings{Khine2017,
        author    = {A. H. Khine and K. T. Nwet and K. M. Soe},
        title     = {Automatic Myanmar News Classification},
        booktitle = {15th Proceedings of International Conference on Computer Applications},
        year      = {2017},
        month     = {February},
        pages     = {401--408}
        }""",
        descriptive_stats={
            "n_samples": {"train": 2048},
            "avg_character_length": {"train": 174.2},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"category": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
