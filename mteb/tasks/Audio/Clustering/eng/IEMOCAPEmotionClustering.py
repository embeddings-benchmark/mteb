from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class IEMOCAPEmotionClustering(AbsTaskAudioClustering):
    label_column_name: str = "major_emotion"

    metadata = TaskMetadata(
        name="IEMOCAPEmotionClustering",
        description="Clustering speech samples by emotion from interactive emotional dyadic conversations in the IEMOCAP database.",
        reference="https://doi.org/10.1007/s10579-008-9076-6",
        dataset={
            "path": "AbstractTTS/IEMOCAP",
            "revision": "9f1696a135a65ce997d898d4121c952269a822ca",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2008-01-01", "2008-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Emotion Clustering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@article{busso2008iemocap,
            title={IEMOCAP: Interactive emotional dyadic motion capture database},
            author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
            journal={Language resources and evaluation},
            volume={42},
            number={4},
            pages={335--359},
            year={2008},
            publisher={Springer}
        }""",
        descriptive_stats={
            "n_samples": {"train": 10039},  # Approximate after subsampling
        },
    )

    audio_column_name: str = "audio"
