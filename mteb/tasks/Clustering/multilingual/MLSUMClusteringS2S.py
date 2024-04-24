from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks import AbsTaskClustering, MultilingualTask, TaskMetadata

_LANGUAGES = {
    "de": ["deu-Latn"],
    "fr": ["fra-Latn"],
    "ru": ["rus-Cyrl"],
    "es": ["spa-Latn"],
}


class MLSUMClusteringS2S(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="MLSUMClusteringS2S",
        description="Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.",
        reference="https://huggingface.co/datasets/mlsum",
        dataset={
            "path": "mlsum",
            "revision": "b5d54f8f3b61ae17845046286940f03c6bc79bc7",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2010-01-01", "2018-09-30"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators=None,
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
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
        """Convert to standard format"""
        _dataset = self.dataset[lang]
        _dataset.pop("train")

        _dataset = _dataset.remove_columns(["summary", "url", "date", "title"])

        for eval_split in self.metadata.eval_splits:
            texts = _dataset[eval_split]["text"]
            topics = _dataset[eval_split]["topic"]
            new_format = {
                "sentences": [split.tolist() for split in np.array_split(texts, 10)],
                "labels": [split.tolist() for split in np.array_split(topics, 10)],
            }
            _dataset[eval_split] = datasets.Dataset.from_dict(new_format)

        self.dataset[lang] = _dataset
