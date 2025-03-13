from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCOPL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-PL",
        description="MS MARCO is a collection of datasets focused on deep learning in search",
        reference="https://microsoft.github.io/msmarco/",
        dataset={
            "path": "clarin-knext/msmarco-pl",
            "revision": "8634c07806d5cce3a6138e260e59b81760a0a640",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-30"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="https://microsoft.github.io/msmarco/",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=""""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        adapted_from=["MSMARCO"],
    )


class MSMARCOPLHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-PLHardNegatives",
        description="MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://microsoft.github.io/msmarco/",
        dataset={
            "path": "mteb/MSMARCO_PL_test_top_250_only_w_correct-v2",
            "revision": "b609cb1ec6772bf92b8e014343a7ecfb10eef2d9",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-30"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="https://microsoft.github.io/msmarco/",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=""""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        adapted_from=["MSMARCO"],
    )
