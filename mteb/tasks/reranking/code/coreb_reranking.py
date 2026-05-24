from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

coreb_description = "CoREB is a contamination-limited, graded-relevance benchmark for evaluating code embedding and reranking models across three retrieval tasks (code2text, text2code, code2code), built from counterfactually rewritten LiveCodeBench problems in five programming languages."

coreb_bibtex = r"""
@article{xue2026coreb,
  author = {Xue, Siqiao and Liao, Zihan and Qin, Jin and Zhang, Ziyin and Mu, Yixiang and Zhou, Fan and Yu, Hang},
  journal = {arXiv preprint arXiv:2605.04615},
  title = {Beyond Retrieval: A Multitask Benchmark and Model for Code Search},
  url = {https://arxiv.org/abs/2605.04615},
  year = {2026},
}
"""

coreb_languages = [
    "eng-Latn",
    "python-Code",
    "ruby-Code",
    "java-Code",
    "go-Code",
    "c++-Code",
]


class CorebC2TReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebC2TReranking",
        description=coreb_description
        + " The Code-to-Text (C2T) task aims to retrieve the problem statement that a code snippet solves.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "mteb/coreb-c2t-reranking",
            "revision": "79beb7ee1b271cfd03966a3f5c553aa2c1cd8a95",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=coreb_languages,
        main_score="map_at_10",
        date=("2026-03-01", "2026-05-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=coreb_bibtex,
        prompt="Retrieve the most relevant problem description for the given code implementation.",
    )


class CorebC2CReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebC2CReranking",
        description=coreb_description
        + " The Code-to-Code (C2C) task aims to retrieve equivalent implementations across different programming languages.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "mteb/coreb-c2c-reranking",
            "revision": "92c6923fc428dd176d4a7379f3dac7564313ae37",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=coreb_languages,
        main_score="map_at_10",
        date=("2026-03-01", "2026-05-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=coreb_bibtex,
        prompt="Retrieve equivalent code implementations across languages.",
    )


class CorebT2CReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebT2CReranking",
        description=coreb_description
        + " The Text-to-Code (T2C) task aims to retrieve code implementations from problem descriptions.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "mteb/coreb-t2c-reranking",
            "revision": "11aaa91e8c4915be01ba73fa89fadd0cebbf4972",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=coreb_languages,
        main_score="map_at_10",
        date=("2026-03-01", "2026-05-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=coreb_bibtex,
        prompt="Retrieve the most relevant code implementation for the given problem description.",
    )
