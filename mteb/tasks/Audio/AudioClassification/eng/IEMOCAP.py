from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class IEMOCAP(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="IEMOCAP",
        description="""IEMOCAP was recorded from ten actors in dyadic sessions with markers on the face, head,
            and hands, which provide detailed information about their facial expression and
            hand movements during scripted and spontaneous spoken communication scenarios.
            actors performed selected emotional scripts and also improvised hypothetical
            scenarios designed to elicit specific types of emotions (happiness, anger, sadness, frustration and neutral state).
            After autmoated annotations, the final emotional categories selected for annotation were :
            anger, sadness, happiness, disgust, fear and surprise, plus frustration, excited and neutral states
        """,
        reference="https://huggingface.co/datasets/AbstractTTS/IEMOCAP",
        dataset={
            "path": "AbstractTTS/IEMOCAP",
            "revision": "9f1696a135a65ce997d898d4121c952269a822ca",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-08-09", "2024-08-11"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="automatic-and-reviewed",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@article{article,
            author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower Provost, Emily and Kim, Samuel and Chang, Jeannette and Lee, Sungbok and Narayanan, Shrikanth},
            year = {2008},
            month = {12},
            pages = {335-359},
            title = {IEMOCAP: Interactive emotional dyadic motion capture database},
            volume = {42},
            journal = {Language Resources and Evaluation},
            doi = {10.1007/s10579-008-9076-6}
        }""",
        # https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Busso_2008_5.pdf
        descriptive_stats={
            "n_samples": {"train": 10039},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True

    def dataset_transform(self):
        ## map labels to ids
        labels = [
            "sad",
            "happy",
            "surprise",
            "frustrated",
            "disgust",
            "angry",
            "neutral",
            "excited",
            "other",
            "fear",
        ]

        label2id = {i: j for j, i in enumerate(labels)}

        self.dataset = self.dataset.map(
            lambda x: {"label": label2id[x["major_emotion"]]}
        )

        ## required to run the dataloader for cross-validation
        import torch

        torch.multiprocessing.set_sharing_strategy("file_system")
        #########################################################
