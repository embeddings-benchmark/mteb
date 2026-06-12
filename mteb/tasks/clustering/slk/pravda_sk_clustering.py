from datasets import Dataset, DatasetDict

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_MAX_DOCUMENT_TO_EMBED = 2048


class PravdaSKTagClustering(AbsTaskClustering):
    max_document_to_embed = _MAX_DOCUMENT_TO_EMBED
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
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify the topic or theme of the given text.",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + summary) and labels (assigned_label)."""
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

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)


class PravdaSKURLClustering(AbsTaskClustering):
    max_document_to_embed = _MAX_DOCUMENT_TO_EMBED
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="PravdaSKURLClustering",
        description="Clustering of Slovak news articles from Pravda.sk based on URL structure. Articles are organized into 50 editorial categories reflecting pravda.sk's content organization, including news, sports, culture, economy, health, travel, celebrity, and science sections.",
        reference="https://huggingface.co/datasets/NaiveNeuron/pravda-sk-url-clustering",
        dataset={
            "path": "NaiveNeuron/pravda-sk-url-clustering",
            "revision": "c5a24605e0fe5a23bc718a531570f353b377e3a3",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify the topic or theme of the given text.",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + summary) and labels (url_category)."""
        ds = {}
        for split in self.metadata.eval_splits:
            # Combine title and summary to create sentences
            titles = self.dataset[split]["title"]
            summaries = self.dataset[split]["summary"]

            sentences = [
                f"{title} {summary}".strip()
                for title, summary in zip(titles, summaries)
            ]

            labels = self.dataset[split]["url_category"]

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)
