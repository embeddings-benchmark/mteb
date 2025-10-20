from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class NSynth(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="NSynth",
        description="Instrument Source Classification: one of acoustic, electronic, or synthetic.",
        reference="https://huggingface.co/datasets/anime-sh/NSYNTH_PITCH_HEAR",
        dataset={
            "path": "mteb/nsynth-mini",
            "revision": "e32dfe9b65e121e64229a821fe1ff177e8962635",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-03-06", "2025-03-06"),
        domains=["Music"],
        task_subtypes=["Instrument Source Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{engel2017neuralaudiosynthesismusical,
  archiveprefix = {arXiv},
  author = {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
  eprint = {1704.01279},
  primaryclass = {cs.LG},
  title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
  url = {https://arxiv.org/abs/1704.01279},
  year = {2017},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "instrument_source"
    samples_per_label: int = 50
