from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoClimateFeverRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoClimateFeverRetrieval",
        description="NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.",
        reference="https://arxiv.org/abs/2012.00614",
        dataset={
            "path": "zeta-alpha-ai/NanoClimateFEVER",
            "revision": "96741bfa30b9f56db8c9eb7d08e775ed6474f206",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Non-fiction", "Academic", "News"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{diggelmann2021climatefever,
      title={CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      author={Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      year={2021},
      eprint={2012.00614},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
        adapted_from=["ClimateFEVER"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoClimateFEVER",
            "corpus",
            revision="96741bfa30b9f56db8c9eb7d08e775ed6474f206",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoClimateFEVER",
            "queries",
            revision="96741bfa30b9f56db8c9eb7d08e775ed6474f206",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoClimateFEVER",
            "qrels",
            revision="96741bfa30b9f56db8c9eb7d08e775ed6474f206",
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
