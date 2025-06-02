from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoNFCorpusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNFCorpusRetrieval",
        description="NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        dataset={
            "path": "zeta-alpha-ai/NanoNFCorpus",
            "revision": "dd542a7efb9ad2136b9e00768b60fca9038f8156",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2016-01-01", "2016-12-31"],
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  city = {Padova},
  country = {Italy},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
  year = {2016},
}
""",
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
        adapted_from=["NFCorpus"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "corpus",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "queries",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "qrels",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
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
