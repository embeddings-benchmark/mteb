from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class BurmeseClustering(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name = "BurmeseClustering", 
        dataset= {
            "path" : "myanmar_news",
            "revision" : "b899ec06227db3679b0fe3c4188a6b48cc0b65eb"
        },
        description="The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.",
        reference = "https://huggingface.co/datasets/myanmar_news",
        type = "Clustering",
        category= "p2p",
        eval_splits= ["train"], 
        eval_langs= ["mya-Mymr"],
        main_score = "accuracy",
        date = ("2017-10-01", "2017-10-31"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Thematic clustering"],
        license="GPL 3.0",
        socioeconomic_status= "low",
        annotations_creators= "derived",
        dialect = [],
        text_creation="found",
        bibtex_citation=""""
        @inproceedings{Khine2017,
        author    = {A. H. Khine and K. T. Nwet and K. M. Soe},
        title     = {Automatic Myanmar News Classification},
        booktitle = {15th Proceedings of International Conference on Computer Applications},
        year      = {2017},
        month     = {February},
        pages     = {401--408}
        }""",
        n_samples={"train": 2048},
        avg_character_length={"train": 174.20059142434698},
    )

    def dataset_transform(self):
        # Rename columns to align with expected names for the processing tasks
        self.dataset = self.dataset.rename_columns({
            "text": "sentences", 
            "category": "label"  # Renaming to 'label' if 'stratified_subsampling' expects this name
        })

        # Call 'stratified_subsampling' assuming it expects the label column named 'label'
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )



