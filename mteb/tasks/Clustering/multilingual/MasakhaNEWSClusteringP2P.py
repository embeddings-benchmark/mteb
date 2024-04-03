from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClustering, MultilingualTask

_LANGUAGES = [
    "amh",
    "eng",
    "fra",
    "hau",
    "ibo",
    "lin",
    "lug",
    "orm",
    "pcm",
    "run",
    "sna",
    "som",
    "swa",
    "tir",
    "xho",
    "yor",
]


class MasakhaNEWSClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="MasakhaNEWSClusteringP2P",
        description="Clustering of news article headlines and texts from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.",
        reference="https://huggingface.co/datasets/masakhane/masakhanews",
        dataset={
            "path": "masakhane/masakhanews",
            "revision": "8ccc72e69e65f40c70e117d8b3c08306bb788b60",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                name=lang,
                **self.metadata_dict["dataset"],
            )
            self.dataset_transform(lang)
        self.data_loaded = True

    def dataset_transform(self, lang):
        """
        Convert to standard format
        """
        self.dataset[lang].pop("train")
        self.dataset[lang].pop("validation")

        self.dataset[lang] = self.dataset[lang].remove_columns(
            ["url", "text", "headline"]
        )
        texts = self.dataset[lang]["test"]["headline_text"]
        labels = self.dataset[lang]["test"]["label"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(texts, 5)],
            "labels": [split.tolist() for split in np.array_split(labels, 5)],
        }
        self.dataset[lang]["test"] = datasets.Dataset.from_dict(new_format)
