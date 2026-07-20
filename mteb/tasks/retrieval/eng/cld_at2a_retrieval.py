from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CLDAT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CLDAT2ARetrieval",
        description=(
            "Composed audio retrieval built from the CLD (Clotho-Difference) subset "
            "of ADIFF. Each query combines a source audio clip with a natural-"
            "language description of how the target differs from it, and the goal "
            "is to retrieve the target clip from the corpus of 1,045 Clotho v2.1 "
            "evaluation recordings. Difference descriptions were generated with a "
            "language-model pipeline by the ADIFF authors. 2,000 queries sampled "
            "from the 5,225 evaluation pairs with a fixed seed."
        ),
        reference="https://arxiv.org/abs/2502.04476",
        dataset={
            "path": "dukesun99/CLD-AT2A",
            "revision": "4db74f3a92fb2e5efad0d35ca5373807b9628c47",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2025-02-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{deshmukh2025adiff,
  author = {Deshmukh, Soham and Han, Shuo and Singh, Rita and Raj, Bhiksha},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  title = {ADIFF: Explaining audio difference using natural language},
  url = {https://arxiv.org/abs/2502.04476},
  year = {2025},
}
""",
        prompt={
            "query": "Given the source audio and a description of how the target differs from it, retrieve the target audio."
        },
    )
