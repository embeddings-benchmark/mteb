from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HumanEvalRetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "HumanEvalRetrieval",
        "description": "HumanEval dataset adapted for code retrieval evaluation.",
        "reference": "https://arxiv.org/abs/2107.03374",
        "dataset": {
            "path": "zeroshot/humaneval-embedding-benchmark",
            "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
            "trust_remote_code": True,
        },
        "type": "Retrieval",
        "category": "s2s",
        "modalities": ["text"],
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "ndcg_at_10",
        "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "mit",
        "annotations_creators": "derived",
        "dialect": [],
        "sample_creation": "found",
        "bibtex_citation": """@article{chen2021evaluating,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and de Oliveira Pinto, Henrique Pon{\'e} and Kaplan, Jared and Edwards, Harri and Burda, Yura and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}""",
        "descriptive_stats": {
            "n_samples": {"test": 158},
            "avg_character_length": {
                "test": {
                    "average_document_length": 491.4240506329114,
                    "average_query_length": 135.9936708860759,
                    "num_documents": 158,
                    "num_queries": 158,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(self.metadata_dict["dataset"]["path"], "corpus")[
            "corpus"
        ]
        queries_ds = load_dataset(self.metadata_dict["dataset"]["path"], "queries")[
            "queries"
        ]
        qrels_ds = load_dataset(self.metadata_dict["dataset"]["path"], "default")[
            "test"
        ]

        # Process corpus
        for item in corpus_ds:
            self.corpus[item["id"]] = {"title": "", "text": item["text"]}

        # Process queries
        for item in queries_ds:
            self.queries[item["id"]] = item["text"]

        # Process qrels (relevant documents)
        for item in qrels_ds:
            query_id = item["query_id"]
            if query_id not in self.relevant_docs:
                self.relevant_docs[query_id] = {}
            self.relevant_docs[query_id][item["corpus_id"]] = item["score"]

        self.data_loaded = True
