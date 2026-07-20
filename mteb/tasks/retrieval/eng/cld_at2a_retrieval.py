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


class CLDA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CLDA2TRetrieval",
        description=(
            "Audio-to-text difference retrieval built from the CLD (Clotho-Difference) "
            "subset of ADIFF. Each query is a Clotho v2.1 audio clip, and the goal is to "
            "retrieve, from a corpus of 2,000 natural-language difference descriptions, "
            "the description(s) that characterise this clip against a paired clip. It is "
            "the audio-to-text inversion of CLDAT2ARetrieval over the same 2,000 "
            "language-model-generated difference captions from the ADIFF authors."
        ),
        reference="https://arxiv.org/abs/2502.04476",
        dataset={
            "path": "dukesun99/CLD-A2T",
            "revision": "cb33bcc92952eeba8c9458be77e97fbe2423677c",
        },
        type="Any2AnyRetrieval",
        category="a2t",
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
            "query": "Given an audio clip, retrieve the description of how it differs from another audio clip."
        },
    )
