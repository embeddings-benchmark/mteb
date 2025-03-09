from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ClusTrecCovidS2SFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ClusTREC-CovidS2S.v2",
        description="A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.",
        reference="https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID",
        dataset={
            "path": "Uri-ka/ClusTREC-Covid-S2S",
            "revision": "b82833f61e1f0f358106b31b3a89cfcb075470cd",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2020-04-10", "2020-07-16"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0", #license",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{katz-etal-2024-knowledge,
            title = "Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature",
            author = "Katz, Uri  and
              Levy, Mosh  and
              Goldberg, Yoav",
            booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
            month = nov,
            year = "2024",
            address = "Miami, Florida, USA",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2024.findings-emnlp.516",
            pages = "8838--8855",
        }
                """,
        prompt="Identify the main category of the covid-19 papers based on the titles and abstracts",
        adapted_from=["ClusTrecCovidP2P"],
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )
            check_label_distribution(self.dataset[split])
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)


class ClusTrecCovidS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ClusTREC-CovidS2S",
        description="A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.",
        reference="https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID",
        dataset={
            "path": "Uri-ka/ClusTREC-Covid-S2S",
            "revision": "b82833f61e1f0f358106b31b3a89cfcb075470cd",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2020-04-10", "2020-07-16"),
        domains=["Academic", "Written", "Medical"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        sample_creation="created",
        bibtex_citation="""@inproceedings{katz-etal-2024-knowledge,
    title = "Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature",
    author = "Katz, Uri  and
      Levy, Mosh  and
      Goldberg, Yoav",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.516",
    pages = "8838--8855",
}
        """,
        prompt="Identify the main category of the covid-19 papers based on the titles and abstracts",
    )


if __name__ == "__main__":
    from mteb.tasks.Clustering import ClusTrecCovidS2S  # Replace with the actual task path

    # Initialize the task
    task = ClusTrecCovidS2S()

    # Compute metadata metrics
    task.calculate_metadata_metrics()

