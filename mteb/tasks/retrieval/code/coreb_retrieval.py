from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"

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


class CorebC2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebC2TRetrieval",
        description=coreb_description
        + " The Code-to-Text (C2T) task aims to retrieve the problem statement that a code snippet solves.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "hq-bench/coreb-c2t-retrieval",
            "revision": "dc306b20b36df58d87171763f7db5bb7ce0f418b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=coreb_languages,
        main_score="ndcg_at_10",
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


class CorebC2CRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebC2CRetrieval",
        description=coreb_description
        + " The Code-to-Code (C2C) task aims to retrieve equivalent implementations across different programming languages.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "hq-bench/coreb-c2c-retrieval",
            "revision": "7fc3acd92e887f8ea4a2eaf724f22cdafe45ec3d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=coreb_languages,
        main_score="ndcg_at_10",
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


class CorebT2CRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CorebT2CRetrieval",
        description=coreb_description
        + " The Text-to-Code (T2C) task aims to retrieve code implementations from problem descriptions.",
        reference="https://arxiv.org/abs/2605.04615",
        dataset={
            "path": "hq-bench/coreb-t2c-retrieval",
            "revision": "227a84ee19db93beeccfff0188299ed95c808742",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=coreb_languages,
        main_score="ndcg_at_10",
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
