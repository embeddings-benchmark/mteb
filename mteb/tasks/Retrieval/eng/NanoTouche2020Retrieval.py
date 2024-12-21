from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoTouche2020Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoTouche2020Retrieval",
        description="NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "zeta-alpha-ai/NanoTouche2020",
            "revision": "0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@dataset{potthast_2022_6862281,
  author       = {Potthast, Martin and
                  Gienapp, Lukas and
                  Wachsmuth, Henning and
                  Hagen, Matthias and
                  Fröbe, Maik and
                  Bondarenko, Alexander and
                  Ajjour, Yamen and
                  Stein, Benno},
  title        = {{Touché20-Argument-Retrieval-for-Controversial- 
                   Questions}},
  month        = jul,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6862281},
  url          = {https://doi.org/10.5281/zenodo.6862281}
}""",
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "corpus",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "queries",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoTouche2020",
            "qrels",
            revision="0d2f26ed8c5ad309f95c7f9499c70a40e140fccd",
        )

        self.corpus = {
            split: {
                sample["_id"]: {"_id": sample["_id"], "text": sample["text"]}
                for sample in self.corpus[split]
            }
            for split in self.corpus
        }

        self.queries = {
            split: {sample["_id"]: sample["text"] for sample in self.queries[split]}
            for split in self.queries
        }

        self.relevant_docs = {
            split: {
                sample["query-id"]: {sample["corpus-id"]: 1}
                for sample in self.relevant_docs[split]
            }
            for split in self.relevant_docs
        }

        self.data_loaded = True
