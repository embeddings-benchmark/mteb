from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class LivedoorNewsClustering(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="LivedoorNewsClustering",
        description="Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics).",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "livedoor_news",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2000-01-01", "2014-02-09"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-nd-2.1-jp",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 1107},
        avg_character_length={"test": 1082.61},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentences", "label": "labels"}
        )
