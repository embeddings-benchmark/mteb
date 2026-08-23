from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ClothoMomentRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClothoMomentRetrieval",
        description=(
            "Language-based Audio Moment Retrieval on the Clotho-Moment dataset. "
            "Given a composed query consisting of a text description of an audio event "
            "and its cropped sound clip, retrieve the correct long background audio "
            "recording containing that moment."
        ),
        reference="https://arxiv.org/abs/2409.15672",
        dataset={
            "path": "deep9539/clotho-moment",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2024-01-01", "2024-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{munakata2025language,
  title={Language-based Audio Moment Retrieval},
  author={Munakata, Hokuto and Nishimura, Taichi and Nakada, Shota and Komatsu, Tatsuya},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
""",
        prompt={
            "query": "Given the background audio and a description of a moment, retrieve the audio segment containing that moment."
        },
    )
