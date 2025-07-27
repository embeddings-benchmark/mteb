from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackTexNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackTex-NL",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a "
        "Dutch-translated version.",
        reference="https://huggingface.co/datasets/clips/beir-nl-cqadupstack",
        dataset={
            "path": "clips/beir-nl-cqadupstack",
            "revision": "1de13232f452f0b4f5525d8c6c41cd6a9ae1d084",
            "split": "tex",
        },
        type="Retrieval",
        category="s2p",
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
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
        adapted_from=["CQADupstackTexRetrieval"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
        )

        qrels_raw = datasets.load_dataset(
            name="test",
            **self.metadata_dict["dataset"],
        )

        self.queries["test"] = {query["_id"]: query["text"] for query in queries_raw}

        self.corpus["test"] = {
            doc["_id"]: doc.get("title", "") + " " + doc["text"] for doc in corpus_raw
        }
        self.relevant_docs["test"] = {
            q["query-id"]: {q["corpus-id"]: int(q["score"])} for q in qrels_raw
        }

        self.data_loaded = True
