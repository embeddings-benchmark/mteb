from __future__ import annotations

import datasets

from mteb.abstasks import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class SwednClusteringFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="SwednClusteringFast",
        dataset={
            "path": "sbx/superlim-2",
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
            "name": "swedn",
            "trust_remote_code": True,
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="p2p",
        eval_splits=["all"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="""@inproceedings{monsen2021method,
  title={A method for building non-english corpora for abstractive text summarization},
  author={Monsen, Julius and J{\"o}nsson, Arne},
  booktitle={Proceedings of CLARIN Annual Conference},
  year={2021}
}""",
        n_samples={"all": 2048},
        avg_character_length={"all": 1619.71},
    )

    def dataset_transform(self):
        """The article_category clusters differ between the splits (with the test set only having 1 cluster). Therefore we combine it all into one
        cluster.
        """
        splits = ["train", "validation"]
        # performance of sample models with test set: 8.74, 2.43 -removing test-> 11.26, 4.27
        # this is due to the test set only having 1 cluster which is "other"

        documents = []
        labels = []
        label_col = "article_category"

        # Note that headlines is not included:
        # Scores of sample models with headlines: 11.26, 4.27 -removing headlines-> 16.43, 4.31
        # as headlines are soo short it is hard to meaningfully cluster them even for humans.
        for split in splits:
            ds_split = self.dataset[split]

            documents.extend(ds_split["summary"])
            labels.extend(ds_split[label_col])

            documents.extend(ds_split["article"])
            labels.extend(ds_split[label_col])
        ds = datasets.Dataset.from_dict({"sentences": documents, "labels": labels})
        self.dataset = datasets.DatasetDict({"all": ds})


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    import mteb
    from mteb.tasks.Clustering.swe.swedn_clustering import SwednClustering

    old_task = SwednClustering()
    task = SwednClusteringFast()
    bench = mteb.MTEB(tasks=[old_task, task])
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    scores = bench.run(model, output_folder="tmp/results")
