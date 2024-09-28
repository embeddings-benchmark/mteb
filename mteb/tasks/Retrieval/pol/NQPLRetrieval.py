from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-PL",
        description="Natural Questions: A Benchmark for Question Answering Research",
        reference="https://ai.google.com/research/NaturalQuestions/",
        dataset={
            "path": "clarin-knext/nq-pl",
            "revision": "f171245712cf85dd4700b06bef18001578d0ca8d",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 502.14302128535564,
                    "average_query_length": 48.31662804171495,
                    "num_documents": 2681468,
                    "num_queries": 3452,
                    "average_relevant_docs_per_query": 1.2169756662804172,
                }
            },
        },
    )


class NQPLHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-PLHardNegatives",
        description="Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        dataset={
            "path": "mteb/NQ_PL_test_top_250_only_w_correct-v2",
            "revision": "9a2878a70ea545a8f4df0cdfa1adea27f4f64390",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": {"test": 1000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 610.7449138094336,
                    "average_query_length": 48.381,
                    "num_documents": 184765,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.213,
                }
            },
        },
    )
