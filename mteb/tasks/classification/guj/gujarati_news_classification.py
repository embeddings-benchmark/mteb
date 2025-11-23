from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GujaratiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GujaratiNewsClassification",
        description="A Gujarati dataset for 3-class classification of Gujarati news articles",
        reference="https://github.com/goru001/nlp-for-gujarati",
        dataset={
            "path": "mlexplorer008/gujarati_news_classification",
            "revision": "1a5f2fa2914bfeff4fcdc6fff4194fa8ec8fa19e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["guj-Gujr"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found
        superseded_by="GujaratiNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headline", "text")


class GujaratiNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GujaratiNewsClassification.v2",
        description="A Gujarati dataset for 3-class classification of Gujarati news articles This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/goru001/nlp-for-gujarati",
        dataset={
            "path": "mteb/gujarati_news",
            "revision": "a815d60b34dc8ee11910743d380b7d14a4e227cb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["guj-Gujr"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found,
        adapted_from=["GujaratiNewsClassification"],
    )
