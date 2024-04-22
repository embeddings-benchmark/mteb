from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = [
    "eng-Latn",
    "pol-Latn",
    "ron-Latn",
    "glg-Latn",
    "lit-Latn",
    "bre-Latn",
    "mlt-Latn",
    "spa-Latn",
    "swa-Latn",
    "slv-Latn",
    "msa-Latn",
    "tgl-Latn",
    "ita-Latn",
    "est-Latn",
    "mlg-Latn",
    "xho-Latn",
    "que-Latn",
    "eus-Latn",
    "nor-Latn",
    "cym-Latn",
    "cat-Latn",
    "fra-Latn",
    "nno-Latn",
    "nld-Latn",
    "ido-Latn",
    "por-Latn",
    "fin-Latn",
    "deu-Latn",
    "dan-Latn",
]


class ToxicChatClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicChatClassification",
        description="""This dataset contains toxicity annotations on 10K user
            prompts collected from the Vicuna online demo. We utilize a human-AI
            collaborative annotation framework to guarantee the quality of annotation
            while maintaining a feasible annotation workload. The details of data
            collection, pre-processing, and annotation can be found in our paper.
            We believe that ToxicChat can be a valuable resource to drive further
            advancements toward building a safe and healthy environment for user-AI
            interactions.
            Only human annotated samples are selected here.""",
        reference="https://aclanthology.org/2023.findings-emnlp.311/",
        dataset={
            "path": "lmsys/toxic-chat",
            "name": "toxicchat0124",
            "revision": "3e0319203c7162b9c9f8015b594441f979c199bc",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="accuracy",
        date=("2023-10-26", "2024-01-31"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 1427},
        avg_character_length={"test": 189.4},
    )

    def dataset_transform(self):
        # only use human-annotated data
        self.dataset = self.dataset["test"].filter(lambda x: x["human_annotation"])
        keep_cols = ["user_input", "toxicity"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [col for col in self.dataset.column_names if col not in keep_cols]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.remove_columns(remove_cols)
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.dataset.train_test_split(
            test_size=0.5, seed=self.seed, stratify_by_column="label"
        )  # balanced sampling across types of hate speech
