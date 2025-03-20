from __future__ import annotations

from collections import Counter

import datasets
import numpy as np

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

NUM_SAMPLES = 2048


class HALClusteringS2S(AbsTaskClustering):
    superseded_by = "HALClusteringS2S.v2"

    metadata = TaskMetadata(
        name="HALClusteringS2S",
        description="Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)",
        reference="https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s",
        dataset={
            "path": "lyon-nlp/clustering-hal-s2s",
            "revision": "e06ebbbb123f8144bef1a5d18796f3dec9ae2915",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{ciancone2024extending,
      title={Extending the Massive Text Embedding Benchmark to French},
      author={Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      year={2024},
      eprint={2405.20468},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
    )

    def dataset_transform(self):
        """Convert to standard format"""
        self.dataset = self.dataset.remove_columns("hal_id")
        titles = self.dataset["test"]["title"]
        domains = self.dataset["test"]["domain"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(titles, 10)],
            "labels": [split.tolist() for split in np.array_split(domains, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)


class HALClusteringS2SFast(AbsTaskClusteringFast):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="HALClusteringS2S.v2",
        description="Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)",
        reference="https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s",
        dataset={
            "path": "lyon-nlp/clustering-hal-s2s",
            "revision": "e06ebbbb123f8144bef1a5d18796f3dec9ae2915",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        date=("2000-03-29", "2024-05-24"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{ciancone2024extending,
      title={Extending the Massive Text Embedding Benchmark to French},
      author={Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      year={2024},
      eprint={2405.20468},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        adapted_from=["HALClusteringS2S"],
    )

    def dataset_transform(self):
        """Convert to standard format"""
        self.dataset["test"] = self.dataset["test"].remove_columns("hal_id")
        self.dataset["test"] = self.dataset["test"].rename_columns(
            {"title": "sentences", "domain": "labels"}
        )
        labels_count = Counter(self.dataset["test"]["labels"])

        # keep classes with more than 2 samples after stratified_subsampling
        frequent_labels = {
            label
            for label, count in labels_count.items()
            if count > len(self.dataset["test"]) * 2 / NUM_SAMPLES
        }
        self.dataset["test"] = self.dataset["test"].filter(
            lambda row: row["labels"] in frequent_labels
        )
        self.dataset["test"] = self.dataset["test"].cast(
            datasets.Features(
                sentences=datasets.Value("string"),
                labels=datasets.ClassLabel(names=sorted(frequent_labels)),
            )
        )
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])

        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
        )
