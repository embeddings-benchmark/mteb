from typing import Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CQADupstackPhysicsNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackPhysics-NL",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a "
        "Dutch-translated version.",
        reference="https://huggingface.co/datasets/clips/beir-nl-cqadupstack",
        dataset={
            "path": "clips/beir-nl-cqadupstack",
            "revision": "1de13232f452f0b4f5525d8c6c41cd6a9ae1d084",
            "split": "physics",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date
        domains=["Written", "Non-fiction"],
        task_subtypes=["Duplicate Detection"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@inproceedings{banar2024beirnlzeroshotinformationretrieval,
  address = {Abu Dhabi, UAE},
  archiveprefix = {arXiv},
  author = {Lotfi, Ehsan and Banar, Nikolay and Daelemans, Walter},
  booktitle = {Proceedings of the 18th Workshop on Building and Using Comparable Corpora ({BUCC})},
  editor = {Sharoff, Serge and Terryn, Ayla Rigouts and Zweigenbaum, Pierre and Rapp, Reinhard},
  eprint = {2412.08329},
  month = {jan},
  pages = {36--45},
  publisher = {Association for Computational Linguistics},
  title = {{BEIR}-{NL}: Zero-shot Information Retrieval Benchmark for the {D}utch Language},
  url = {https://aclanthology.org/2025.bucc-1.5/},
  year = {2025},
}
""",
        adapted_from=["CQADupstackPhysicsRetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            **self.metadata.dataset,
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata.dataset,
        )

        qrels_raw = datasets.load_dataset(
            name="test",
            **self.metadata.dataset,
        )

        self.queries["test"] = {query["_id"]: query["text"] for query in queries_raw}

        self.corpus["test"] = {
            doc["_id"]: {
                "text": doc["text"],
                "title": doc["title"],
            }
            for doc in corpus_raw
        }
        self.relevant_docs["test"] = {
            q["query-id"]: {q["corpus-id"]: int(q["score"])} for q in qrels_raw
        }

        self.data_loaded = True
