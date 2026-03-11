import datasets
import numpy as np

from mteb.abstasks.clustering import (
    AbsTaskClustering,
)
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

NUM_SAMPLES = 2048


class HALClusteringS2S(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="HALClusteringS2S",
        description="Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)",
        reference="https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s",
        dataset={
            "path": "lyon-nlp/clustering-hal-s2s",
            "revision": "e06ebbbb123f8144bef1a5d18796f3dec9ae2915",
        },
        type="Clustering",
        category="t2c",
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
        bibtex_citation=r"""
@misc{ciancone2024extending,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {Extending the Massive Text Embedding Benchmark to French},
  year = {2024},
}
""",
        superseded_by="HALClusteringS2S.v2",
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        """Convert to standard format"""
        self.dataset = self.dataset.remove_columns("hal_id")
        titles = self.dataset["test"]["title"]
        domains = self.dataset["test"]["domain"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(titles, 10)],
            "labels": [split.tolist() for split in np.array_split(domains, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)


class HALClusteringS2SFast(AbsTaskClustering):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="HALClusteringS2S.v2",
        description="Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)",
        reference="https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s",
        dataset={
            "path": "mteb/HALClusteringS2S.v2",
            "revision": "f1b4d7e7d58992005012c43c9dd2cc7436a1e378",
        },
        type="Clustering",
        category="t2c",
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
        bibtex_citation=r"""
@misc{ciancone2024extending,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {Extending the Massive Text Embedding Benchmark to French},
  year = {2024},
}
""",
        adapted_from=["HALClusteringS2S"],
    )
