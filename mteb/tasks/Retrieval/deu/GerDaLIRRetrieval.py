from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class GerDaLIR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GerDaLIR",
        description="GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "jinaai/ger_da_lir",
            "revision": "0bb47f1d73827e96964edb84dfe552f62f4fd5eb",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{wrzalik-krechel-2021-gerdalir,
    title = "{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval",
    author = "Wrzalik, Marco  and
      Krechel, Dirk",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.13",
    pages = "123--128",
    abstract = "We present GerDaLIR, a German Dataset for Legal Information Retrieval based on case documents from the open legal information platform Open Legal Data. The dataset consists of 123K queries, each labelled with at least one relevant document in a collection of 131K case documents. We conduct several baseline experiments including BM25 and a state-of-the-art neural re-ranker. With our dataset, we aim to provide a standardized benchmark for German LIR and promote open research in this area. Beyond that, our dataset comprises sufficient training data to be used as a downstream task for German or multilingual language models.",
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(
            name="queries",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        corpus_rows = datasets.load_dataset(
            name="corpus",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        qrels_rows = datasets.load_dataset(
            name="qrels",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        self.queries = {
            self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}
        }
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows
            }
        }

        self.data_loaded = True
