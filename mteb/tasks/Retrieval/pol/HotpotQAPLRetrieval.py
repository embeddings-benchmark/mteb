from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQAPL(AbsTaskRetrieval):
    metadata = TaskMetadata.model_construct(
        name="HotpotQA-PL",
        description="HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.",
        reference="https://hotpotqa.github.io/",
        dataset={
            "path": "mteb/HotpotQA-PL",
            "revision": "49835d4dc7312e5d8df554462438f0b65c8ef6a6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["HotpotQA"],
    )


class HotpotQAPLHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata.model_construct(
        name="HotpotQA-PLHardNegatives",
        description="HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://hotpotqa.github.io/",
        dataset={
            "path": "mteb/HotpotQA_PL_test_top_250_only_w_correct-v2",
            "revision": "0642cadffa3205c6b21c9af901fdffcd60d6f31e",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["HotpotQA"],
    )
