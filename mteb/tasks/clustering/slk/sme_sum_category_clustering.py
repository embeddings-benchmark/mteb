from datasets import Dataset, DatasetDict

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class SMESumCategoryClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="SMESumCategoryClustering",
        description="Clustering of Slovak news articles from SMESum dataset based on news categories. Articles are organized into 11 thematic categories covering various topics from the SME news portal including politics, economy, sports, culture, and other news domains. Articles with 'none' category are excluded.",
        reference="https://aclanthology.org/2020.lrec-1.830/",
        dataset={
            "path": "NaiveNeuron/SMESum",
            "revision": "c5a6521a4ddce3450fb04ba218623681a9189c6d",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2013-01-01", "2019-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{suppa-adamec-2020-summarization,
  address = {Marseille, France},
  author = {Suppa, Marek and Adamec, Jergus},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta and B{\'e}chet, Fr{\'e}d{\'e}ric and Blache, Philippe and Choukri, Khalid and Cieri, Christopher and Declerck, Thierry and Goggi, Sara and Isahara, Hitoshi and Maegaard, Bente and Mariani, Joseph and Mazo, H{\'e}l{\`e}ne and Moreno, Asuncion and Odijk, Jan and Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {6725--6730},
  publisher = {European Language Resources Association},
  title = {A Summarization Dataset of {S}lovak News Articles},
  url = {https://aclanthology.org/2020.lrec-1.830/},
  year = {2020},
}
""",
        prompt="Identify the topic or theme of the given text.",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + introduction) and labels (category).

        Filters out articles with 'none' category as they don't represent meaningful clusters.
        """
        ds = {}
        for split in self.metadata.eval_splits:
            titles = self.dataset[split]["title"]
            introductions = self.dataset[split]["introduction"]
            categories = self.dataset[split]["category"]

            sentences = []
            labels = []
            for title, introduction, category in zip(titles, introductions, categories):
                if category is not None and category != "none":
                    sentences.append(f"{title} {introduction}".strip())
                    labels.append(category)

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)
