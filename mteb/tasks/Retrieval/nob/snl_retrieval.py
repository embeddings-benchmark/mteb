from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SNLRetrieval",
        dataset={
            "path": "navjordj/SNL_summarization",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
        },
        description="Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.",
        reference="https://huggingface.co/datasets/navjordj/SNL_summarization",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version assumed (not specified beforehand)
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1986.9453846153847,
                    "average_query_length": 14.906153846153845,
                    "num_documents": 1300,
                    "num_queries": 1300,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
        task_subtypes=["Article retrieval"],
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = dict[doc_id, dict[str, str]] #id => dict with document datas like title and text
        self.queries = dict[query_id, str] #id => query
        self.relevant_docs = dict[query_id, dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)

            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            headline = ds["headline"]
            article = ds["article"]

            n = 0
            for headl, art in zip(headline, article):
                self.queries[split][str(n)] = headl
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1
                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[art]): 1
                }  # only one correct matches
