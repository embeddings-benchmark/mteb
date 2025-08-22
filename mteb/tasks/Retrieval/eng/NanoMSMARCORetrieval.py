from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoMSMARCORetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoMSMARCORetrieval",
        description="NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.",
        reference="https://microsoft.github.io/msmarco/",
        dataset={
            "path": "zeta-alpha-ai/NanoMSMARCO",
            "revision": "7b8ff22f2771dc65ac5b439f222eb19a1f56abda",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2016-01-01", "2016-12-31"],
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/NguyenRSGTMD16,
  archiveprefix = {arXiv},
  author = {Tri Nguyen and
Mir Rosenberg and
Xia Song and
Jianfeng Gao and
Saurabh Tiwary and
Rangan Majumder and
Li Deng},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  eprint = {1611.09268},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  url = {http://arxiv.org/abs/1611.09268},
  volume = {abs/1611.09268},
  year = {2016},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
        adapted_from=["MSMARCO"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoMSMARCO",
            "corpus",
            revision="7b8ff22f2771dc65ac5b439f222eb19a1f56abda",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoMSMARCO",
            "queries",
            revision="7b8ff22f2771dc65ac5b439f222eb19a1f56abda",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoMSMARCO",
            "qrels",
            revision="7b8ff22f2771dc65ac5b439f222eb19a1f56abda",
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

        relevant_docs = {}

        for split in self.relevant_docs:
            relevant_docs[split] = defaultdict(dict)
            for query_id, corpus_id in zip(
                self.relevant_docs[split]["query-id"],
                self.relevant_docs[split]["corpus-id"],
            ):
                relevant_docs[split][query_id][corpus_id] = 1
        self.relevant_docs = relevant_docs

        self.data_loaded = True
