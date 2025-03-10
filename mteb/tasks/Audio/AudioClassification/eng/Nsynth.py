from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class NSynth(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="NSynth",
        description="Instrument Source Classification: one of acoustic, electronic, or synthetic.",
        reference="https://huggingface.co/datasets/anime-sh/NSYNTH_PITCH_HEAR",
        dataset={
            "path": "anime-sh/NSYNTH_PITCH_HEAR",
            "revision": "6e39b6b61d86d416e591230525e234cc0a5b753a",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-03-06", "2025-03-06"),
        domains=[],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Instrument Source Classification"],
        license="not specified",  # Replace with appropriate license from allowed list
        annotations_creators="...",
        dialect=[],
        modalities=["audio"],
        sample_creation="...",
        bibtex_citation="""@misc{engel2017neuralaudiosynthesismusical,
            title={Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders}, 
            author={Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
            year={2017},
            eprint={1704.01279},
            archivePrefix={arXiv},
            primaryClass={cs.LG},
            url={https://arxiv.org/abs/1704.01279}, 
        }""",
        descriptive_stats={
            "n_samples": {"train": 289000, "validation": 12700, "test": 4100},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "instrument_source"
    samples_per_label: int = 50
