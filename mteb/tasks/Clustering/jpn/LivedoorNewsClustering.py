from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class LivedoorNewsClusteringv2(AbsTaskClusteringFast):
    max_document_to_embed = 1107
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="LivedoorNewsClustering.v2",
        description="Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics). Version 2 updated on LivedoorNewsClustering by removing pairs where one of entries contain an empty sentences.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "livedoor_news",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2000-01-01", "2014-02-09"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nd-2.1-jp",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 1106},
            "avg_character_length": {"test": 1082.61},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentences", "label": "labels"}
        )

        for split in self.metadata.eval_splits:
            # remove empty sentences (there is only one per split)
            self.dataset[split] = self.dataset[split].filter(
                lambda x: len(x["sentences"]) > 0
            )


class LivedoorNewsClustering(AbsTaskClusteringFast):
    max_document_to_embed = 1107
    max_fraction_of_documents_to_embed = None
    superseded_by = "LivedoorNewsClustering.v2"

    metadata = TaskMetadata(
        name="LivedoorNewsClustering",
        description="Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics).",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "livedoor_news",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2000-01-01", "2014-02-09"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nd-2.1-jp",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 1107},
            "avg_character_length": {"test": 1082.61},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentences", "label": "labels"}
        )
