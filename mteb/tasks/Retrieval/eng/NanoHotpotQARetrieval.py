from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoHotpotQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoHotpotQARetrieval",
        description="NanoHotpotQARetrieval is a smaller subset of the "
        + "HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong"
        + " supervision for supporting facts to enable more explainable question answering systems.",
        reference="https://hotpotqa.github.io/",
        dataset={
            "path": "zeta-alpha-ai/NanoHotpotQA",
            "revision": "d79c0cdda980aba54842756770928035e1b61a51",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-12-31"],
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{yang-etal-2018-hotpotqa,
  abstract = {Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.},
  address = {Brussels, Belgium},
  author = {Yang, Zhilin  and
Qi, Peng  and
Zhang, Saizheng  and
Bengio, Yoshua  and
Cohen, William  and
Salakhutdinov, Ruslan  and
Manning, Christopher D.},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1259},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {2369--2380},
  publisher = {Association for Computational Linguistics},
  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  url = {https://aclanthology.org/D18-1259},
  year = {2018},
}
""",
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
        adapted_from=["HotpotQA"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoHotpotQA",
            "corpus",
            revision="d79c0cdda980aba54842756770928035e1b61a51",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoHotpotQA",
            "queries",
            revision="d79c0cdda980aba54842756770928035e1b61a51",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoHotpotQA",
            "qrels",
            revision="d79c0cdda980aba54842756770928035e1b61a51",
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
