from __future__ import annotations

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
        bibtex_citation="""@article{DBLP:journals/corr/NguyenRSGTMD16,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  journal   = {CoRR},
  volume    = {abs/1611.09268},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.09268},
  archivePrefix = {arXiv},
  eprint    = {1611.09268},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
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

        self.relevant_docs = {
            split: {
                sample["query-id"]: {sample["corpus-id"]: 1}
                for sample in self.relevant_docs[split]
            }
            for split in self.relevant_docs
        }

        self.data_loaded = True
