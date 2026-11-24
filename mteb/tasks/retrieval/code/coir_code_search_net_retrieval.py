from __future__ import annotations

import logging

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


class COIRCodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="COIRCodeSearchNetRetrieval",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.",
        reference="https://huggingface.co/datasets/code_search_net/",
        dataset={
            "path": "mteb/COIRCodeSearchNetRetrieval",
            "revision": "667a2e235be1bae585b8e221d038f0b455ecd825",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{husain2019codesearchnet,
  author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal = {arXiv preprint arXiv:1909.09436},
  title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
  year = {2019},
}
""",
    )
