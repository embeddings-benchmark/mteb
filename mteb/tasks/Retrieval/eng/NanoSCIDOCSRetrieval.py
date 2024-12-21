from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoSCIDOCSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSCIDOCSRetrieval",
        description="NanoFiQA2018 is a smaller subset of "
        + "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
        + " prediction, to document classification and recommendation.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "zeta-alpha-ai/NanoSCIDOCS",
            "revision": "484eb90549fc3f0b9c42b3551e80ceb999515537",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
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
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "corpus",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "queries",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "qrels",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
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
