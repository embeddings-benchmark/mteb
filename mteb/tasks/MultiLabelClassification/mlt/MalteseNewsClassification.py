from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MalteseNewsClassification(AbsTaskMultilabelClassification):
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
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["mlt-Latn"],
        main_score="accuracy",
        date=("2023-10-21", "2024-04-24"),
        domains=["Constructed", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{maltese-news-datasets,
  author = {Chaudhary, Amit Kumar  and
Micallef, Kurt  and
Borg, Claudia},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  month = may,
  publisher = {Association for Computational Linguistics},
  title = {Topic Classification and Headline Generation for {M}altese using a Public News Corpus},
  year = {2024},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"labels": "label"})
        remove_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(remove_cols)
