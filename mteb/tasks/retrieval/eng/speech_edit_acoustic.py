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
        reference="https://arxiv.org/abs/2606.01804",
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
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{zhang2026speecheditbench,
  author = {Zhang, Hanlin and Tan, Daxin and Tao, Dehua and Chen, Xiao and Tan, Haochen and Song, Linqi},
  journal = {arXiv preprint arXiv:2606.01804},
  title = {SpeechEditBench: A Bilingual Multi-Attribute Benchmark for Instruction-Guided Speech Editing},
  year = {2026},
}
""",
        prompt={
            "query": "Given the source audio and an editing instruction, retrieve the corresponding edited audio."
        },
    )
