import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.clustering import (
    AbsTaskClustering,
    _check_label_distribution,
)
from mteb.abstasks.task_metadata import TaskMetadata


class BeytooteClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BeytooteClustering",
        description="Beytoote Website Articles Clustering",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/beytoote-clustering",
            "revision": "62ca5aecb9414214162569f2f1bfb07aa219a70e",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="v_measure",
        date=("2024-09-01", "2024-12-31"),
        domains=["News"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            label="labels",
        )


class DigikalamagClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="DigikalamagClustering",
        description="A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.",
        reference="https://hooshvare.github.io/docs/datasets/tc",
        dataset={
            "path": "mteb/DigikalamagClustering",
            "revision": "0fe394ee57514d4dbc3deeb1b1ae5dbd8cb7e52b",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="v_measure",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class HamshahriClustring(AbsTaskClustering):
    metadata = TaskMetadata(
        name="HamshahriClustring",
        description="These datasets have been extracted from the RSS feed of two Farsi news agency websites.",
        reference="https://github.com/mallahyari/Farsi-datasets",
        dataset={
            "path": "community-datasets/farsi_news",
            "revision": "ca93dc707cea06cdb2e3ede3b547a1092053aca6",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="v_measure",
        date=("2024-09-01", "2024-12-31"),
        domains=["News"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.map(
            lambda x: {"sentences": f"{x['title']}\n: {x['summary']}"}
        )
        self.dataset = self.dataset.map(lambda x: {"labels": x["tags"][0]})
        self.dataset = DatasetDict({"test": self.dataset["hamshahri"]})

        ds = {}
        for split in self.metadata.eval_splits:
            labels = self.dataset[split]["labels"]
            sentences = self.dataset[split]["sentences"]

            _check_label_distribution(self.dataset[split])

            # Remove sentences and labels with only 1 label example.
            unique_labels, counts = np.unique(labels, return_counts=True)
            solo_label_idx = np.where(counts == 1)
            solo_labels = unique_labels[solo_label_idx]
            is_solo = np.isin(labels, solo_labels)
            split_ds = Dataset.from_dict({"labels": labels, "sentences": sentences})
            if is_solo.any():
                split_ds = split_ds.select(np.nonzero(is_solo == False)[0])  # noqa: E712
            ds[split] = split_ds
        self.dataset = DatasetDict(ds)

        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            label="labels",
        )


class NLPTwitterAnalysisClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="NLPTwitterAnalysisClustering",
        description="Clustering of tweets from twitter across 26 categories.",
        reference="https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/commits/main",
        dataset={
            "path": "hamedhf/nlp_twitter_analysis",
            "revision": "4ceb1312583fd2c7c73ad2d550b726124dcd39a0",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="v_measure",
        date=("2024-09-01", "2024-12-31"),
        domains=["Social"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("tweet", "sentences")
        self.dataset = self.dataset.rename_column("label", "labels")
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            label="labels",
        )


class SIDClustring(AbsTaskClustering):
    metadata = TaskMetadata(
        name="SIDClustring",
        description="Clustering of summariesfrom SIDClustring across categories.",
        reference="https://www.sid.com/",
        dataset={
            "path": "MCINext/sid-clustering",
            "revision": "d361bb18535d592e845aeaaa8ac47a239aa2f87a",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="v_measure",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            label="labels",
        )
