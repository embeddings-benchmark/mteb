from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks import AbsTaskClustering, MultilingualTask
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "bs": ["bos-Latn"],
    "ca": ["cat-Latn"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "eu": ["eus-Latn"],
    "gv": ["glv-Latn"],
    "ilo": ["ilo-Latn"],
    "ku": ["kur-Latn"],
    "lv": ["lav-Latn"],
    "min": ["min-Latn"],
    "mt": ["mlt-Latn"],
    "sco": ["sco-Latn"],
    "sq": ["sqi-Latn"],
    "wa": ["wln-Latn"],
}


class WikiClusteringP2P(AbsTaskClustering, MultilingualTask):
    superseeded_by = "WikiClusteringFastP2P"
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation=None,  # None exists
        n_samples={"test": 71680},
        avg_character_length={"test": 625.3},
    )


class WikiClusteringFastP2P(AbsTaskClusteringFast, MultilingualTask):
    metadata = TaskMetadata(
        name="WikiClusteringFastP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",  # None exists
        n_samples={"test": 2048},
        avg_character_length={"test": 625.3},
    )

    def dataset_transform(self):
        ds = dict()
        for lang in self.hf_subsets:
            labels = []
            sentences = []
            ds[lang] = dict()
            lang_dict = dict()
            for split in self.metadata.eval_splits:
                labels.extend(
                    list(
                        itertools.chain.from_iterable(
                            self.dataset[lang][split]["labels"]
                        )
                    )
                )
                sentences.extend(
                    list(
                        itertools.chain.from_iterable(
                            self.dataset[lang][split]["sentences"]
                        )
                    )
                )

                # Remove sentences and labels with only 1 label example.
                unique_labels, counts = np.unique(labels, return_counts=True)
                solo_label_idx = np.where(counts == 1)
                solo_labels = unique_labels[solo_label_idx]
                for solo_label in solo_labels:
                    loc = labels.index(solo_label)
                    labels.pop(loc)
                    sentences.pop(loc)

                lang_dict.update(
                    {
                        split: Dataset.from_dict(
                            {"labels": labels, "sentences": sentences}
                        )
                    }
                )
            ds[lang] = DatasetDict(lang_dict)
        self.dataset = DatasetDict(ds)
        for lang in self.hf_subsets:
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang],
                self.seed,
                self.metadata.eval_splits,
                label="labels",
                n_samples=2048,
            )
