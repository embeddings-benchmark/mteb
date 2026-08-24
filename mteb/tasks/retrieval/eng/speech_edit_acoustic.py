from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SpeechEditAcousticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpeechEditAcousticRetrieval",
        description=(
            "Composed audio retrieval task based on the acoustic_editing subset of DiscreteSpeech/SpeechEditBench. "
            "Each query combines an original speech recording with a natural-language editing instruction, "
            "and the goal is to retrieve the corresponding edited target recording."
        ),
        reference="https://huggingface.co/datasets/DiscreteSpeech/SpeechEditBench",
        dataset={
            "path": "deep9539/speech_edit_acoustic",
            "revision": "27955b1e1dfc1b433602a26f697a9d19de710d36",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2024-01-01", "2026-08-23"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{speecheditbench2024,
  title={SpeechEditBench: A Benchmark for Speech Editing},
  author={DiscreteSpeech},
  year={2024},
  url={https://huggingface.co/datasets/DiscreteSpeech/SpeechEditBench}
}
""",
        prompt={
            "query": "Given the source audio and an editing instruction, retrieve the corresponding edited audio."
        },
    )
