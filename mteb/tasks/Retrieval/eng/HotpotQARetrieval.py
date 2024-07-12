from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA",
        dataset={
            "path": "mteb/hotpotqa",
            "revision": "ab518f4d6fcca38d87c25209f94beba119d02014",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            " supervision for supporting facts to enable more explainable question answering systems."
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{yang-etal-2018-hotpotqa,
    title = "{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering",
    author = "Yang, Zhilin  and
      Qi, Peng  and
      Zhang, Saizheng  and
      Bengio, Yoshua  and
      Cohen, William  and
      Salakhutdinov, Ruslan  and
      Manning, Christopher D.",
    editor = "Riloff, Ellen  and
      Chiang, David  and
      Hockenmaier, Julia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1259",
    doi = "10.18653/v1/D18-1259",
    pages = "2369--2380",
    abstract = "Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.",
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "train": {
                    "average_document_length": 287.9079517072212,
                    "average_query_length": 105.54965882352941,
                    "num_documents": 5233329,
                    "num_queries": 85000,
                    "average_relevant_docs_per_query": 2.0,
                },
                "dev": {
                    "average_document_length": 287.9079517072212,
                    "average_query_length": 105.35634294106848,
                    "num_documents": 5233329,
                    "num_queries": 5447,
                    "average_relevant_docs_per_query": 2.0,
                },
                "test": {
                    "average_document_length": 287.9079517072212,
                    "average_query_length": 92.17096556380824,
                    "num_documents": 5233329,
                    "num_queries": 7405,
                    "average_relevant_docs_per_query": 2.0,
                },
            },
        },
    )
