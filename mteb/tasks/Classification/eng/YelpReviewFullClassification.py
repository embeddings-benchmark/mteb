from __future__ import annotations

import datasets
import pandas as pd

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class YelpReviewFullClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YelpReviewFullClassification",
        description="Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "yelp_review_full",
            "revision": "c1f9ee939b7d05667af864ee1cb066393154bf85",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        form=["written"],
        domains=["Reviews"],
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 50000},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        total_rows = 2048
        eval_splits = self.metadata.eval_splits

        ds = {"train": self.dataset["train"]}

        for split in eval_splits:
            ds_split = self.dataset[split]

            df_split = pd.DataFrame(
                {"text": ds_split["text"], "label": ds_split["label"]}
            )

            label_counts = df_split["label"].value_counts()
            total_labels = len(label_counts)
            rows_per_label = total_rows // total_labels

            sampled_dfs = []
            for label, count in label_counts.items():
                sampled_df = df_split[df_split["label"] == label].sample(
                    n=min(rows_per_label, count), random_state=42
                )
                sampled_dfs.append(sampled_df)

            sampled_df = pd.concat(sampled_dfs, ignore_index=True)

            ds[split] = datasets.Dataset.from_dict(
                {
                    "text": sampled_df["text"].tolist(),
                    "label": sampled_df["label"].tolist(),
                }
            )

        self.dataset = datasets.DatasetDict(ds)
