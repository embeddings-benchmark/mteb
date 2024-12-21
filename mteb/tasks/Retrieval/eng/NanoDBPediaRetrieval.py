from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoDBPediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPediaRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.",
        reference="https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia",
        dataset={
            "path": "zeta-alpha-ai/NanoDBPedia",
            "revision": "438f1c25129f05db6238699b5afdc9c6b58d2096",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2015-01-01", "2015-12-31"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{lehmann2015dbpedia, title={DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia}, author={Lehmann, Jens and et al.}, journal={Semantic Web}, year={2015}}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "corpus",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "queries",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "qrels",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
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
