from __future__ import annotations

from collections import Counter

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MalteseNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalteseNewsClassification",
        description="""A multi-label topic classification dataset for Maltese News
        Articles. The data was collected from the press_mt subset from Korpus
        Malti v4.0. Article contents were cleaned to filter out JavaScript, CSS,
        & repeated non-Maltese sub-headings. The labels are based on the category
        field from this corpus. 
        """,
        reference="https://huggingface.co/datasets/MLRS/maltese_news_categories",
        dataset={
            "path": "MLRS/maltese_news_categories",
            "revision": "6bb0321659c4f07c4c2176c30c98c971be6571b4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["mlt-Latn"],
        main_score="accuracy",
        date=("2023-10-21", "2024-04-24"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{maltese-news-datasets,
            title = "Topic Classification and Headline Generation for {M}altese using a Public News Corpus",
            author = "Chaudhary, Amit Kumar  and
                    Micallef, Kurt  and
                    Borg, Claudia",
            booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
            month = may,
            year = "2024",
            publisher = "Association for Computational Linguistics",
        }""",
        n_samples={"train": 1146, "test": 1146},
        avg_character_length={"train": 1710.05, "test": 1797.47},
    )

    def dataset_transform(self):
        # 80% of categories have just one label, so it's safe to take the first
        self.dataset = self.dataset["test"].map(lambda x: {"labels" : x["labels"][0]})
        rename_dict = dict(zip(["labels"], ["label"]))
        self.dataset = self.dataset.rename_columns(rename_dict)
        remove_cols = [col for col in self.dataset.column_names if col not in ["text", "label"]]
        self.dataset = self.dataset.remove_columns(remove_cols)
        self.dataset = self.dataset.class_encode_column("label")
        # find low count labels with lower than 6 occurances
        label_counts = Counter(self.dataset['label'])
        low_count_labels = [k for k,v in label_counts.items() if v < 6]
        self.dataset = self.dataset.filter(lambda x: not x['label'] in low_count_labels)
        # split dataset
        self.dataset = self.dataset.train_test_split(
            test_size=0.5, seed=self.seed, stratify_by_column="label"
        )
