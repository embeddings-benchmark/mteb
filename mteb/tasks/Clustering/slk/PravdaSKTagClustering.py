from __future__ import annotations

from datasets import Dataset, DatasetDict

from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from ....abstasks.TaskMetadata import TaskMetadata


class PravdaSKTagClustering(AbsTaskClusteringFast):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="PravdaSKTagClustering",
        description="Clustering of Slovak news articles from Pravda.sk based on article tags. Articles are grouped into 50 thematic categories including Slovak politics, international affairs, events, and topics.",
        reference="https://huggingface.co/datasets/NaiveNeuron/pravda-sk-tag-clustering",
        dataset={
            "path": "NaiveNeuron/pravda-sk-tag-clustering",
            "revision": "dd0a6c077151b8c8bc2fd6abcd746b34fde80bf8",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not-specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify topic categories based on article tags in Slovak news",
    )

    def dataset_transform(self):
        """
        Transform the dataset to create sentences (title + summary) and labels (assigned_label).
        """
        ds = {}
        for split in self.metadata.eval_splits:
            # Combine title and summary to create sentences
            titles = self.dataset[split]["title"]
            summaries = self.dataset[split]["summary"]

            sentences = [
                f"{title} {summary}".strip()
                for title, summary in zip(titles, summaries)
            ]

            labels = self.dataset[split]["assigned_label"]

            ds[split] = Dataset.from_dict({
                "sentences": sentences,
                "labels": labels
            })

        self.dataset = DatasetDict(ds)
