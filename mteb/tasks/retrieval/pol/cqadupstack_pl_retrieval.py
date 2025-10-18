from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CQADupstackWordpressRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Wordpress-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl",
        dataset={
            "path": "mteb/CQADupstack-Wordpress-PL",
            "revision": "b93d34a19cea03d6c7b00e261eccace6b29535ee",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Web", "Programming"],
        task_subtypes=["Question answering"],
        license="not specified",
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
        adapted_from=["CQADupstackWordpressRetrieval"],
    )


class CQADupstackWebmastersRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Webmasters-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl",
        dataset={
            "path": "mteb/CQADupstack-Webmasters-PL",
            "revision": "aca4f7fb71a6d367e1c8f5e3fda118a4bcdb41d3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Web"],
        task_subtypes=["Question answering"],
        license="not specified",
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
        adapted_from=["CQADupstackWebmastersRetrieval"],
    )


class CQADupstackUnixRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Unix-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl",
        dataset={
            "path": "mteb/CQADupstack-Unix-PL",
            "revision": "f41e1fefa91ccb25964afb24e193d1e2a14bb4a3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Web", "Programming"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackUnixRetrieval"],
    )


class CQADupstackTexRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Tex-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl",
        dataset={
            "path": "mteb/CQADupstack-Tex-PL",
            "revision": "7e24a1d7543917b655899680f7e03fed3fe967d4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackTexRetrieval"],
    )


class CQADupstackStatsRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Stats-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl",
        dataset={
            "path": "mteb/CQADupstack-Stats-PL",
            "revision": "c1fac8ba4c0f1ed59910f5278c6bf071008b0180",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Academic", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackStatsRetrieval"],
    )


class CQADupstackProgrammersRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Programmers-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl",
        dataset={
            "path": "mteb/CQADupstack-Programmers-PL",
            "revision": "6536329882b0f06bf6406fdafedc434603fe8ed9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Programming", "Written", "Non-fiction"],
        task_subtypes=[],
        license="not specified",
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
        adapted_from=["CQADupstackProgrammersRetrieval"],
    )


class CQADupstackPhysicsRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Physics-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl",
        dataset={
            "path": "mteb/CQADupstack-Physics-PL",
            "revision": "a61c9ba627e0fa44b85578812577e6665a60661c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Academic", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackPhysicsRetrieval"],
    )


class CQADupstackMathematicaRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Mathematica-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl",
        dataset={
            "path": "mteb/CQADupstack-Mathematica-PL",
            "revision": "f8c9c441534d4823bf9b06573e239841900dec3e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Academic", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackMathematicaRetrieval"],
    )


class CQADupstackGisRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Gis-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl",
        dataset={
            "path": "mteb/CQADupstack-Gis-PL",
            "revision": "784afb42ad5c0f2e71630644a32f08645bbc87fa",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written", "Academic", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackGisRetrieval"],
    )


class CQADupstackGamingRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Gaming-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl",
        dataset={
            "path": "mteb/CQADupstack-Gaming-PL",
            "revision": "7ae0bc1b9a8a1d5a8587a464e25c853552302ce7",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Web", "Written"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackGamingRetrieval"],
    )


class CQADupstackEnglishRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-English-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl",
        dataset={
            "path": "mteb/CQADupstack-English-PL",
            "revision": "27402a7ed3c8a51f45f2a84cb7dfaf6ab4729dad",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Written"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackEnglishRetrieval"],
    )


class CQADupstackAndroidRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Android-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl",
        dataset={
            "path": "mteb/CQADupstack-Android-PL",
            "revision": "1f7d469f28e94c23164476fc4115f14fb3b95a6c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date,
        domains=["Programming", "Web", "Written", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="not specified",
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
        adapted_from=["CQADupstackAndroidRetrieval"],
    )
