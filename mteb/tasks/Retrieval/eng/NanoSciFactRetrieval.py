from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoSciFactRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSciFactRetrieval",
        description="NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        dataset={
            "path": "zeta-alpha-ai/NanoSciFact",
            "revision": "309f1d1ae3ae2e092444a8a0c25bed59b82318bc",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-12-31"],
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoSciFact",
            "corpus",
            revision="309f1d1ae3ae2e092444a8a0c25bed59b82318bc",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoSciFact",
            "queries",
            revision="309f1d1ae3ae2e092444a8a0c25bed59b82318bc",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoSciFact",
            "qrels",
            revision="309f1d1ae3ae2e092444a8a0c25bed59b82318bc",
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
