from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskZeroshotAudioClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class CREMADZeroshotClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="CREMADZeroshot",
        description="Classifying 6 emotions in actor's voice recordings of same text spoken in different emotions",
        reference="https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/",
        dataset={
            "path": "AbstractTTS/CREMA-D"
        },
        type="ZeroShotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-latn"],
        main_score="accuracy",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="odbl-1.0",
        modalities=["audio", "text"],
        sample_creation="created",
        bibtex_citation="""@ARTICLE{Cao2014-ih,
  title     = "{CREMA-D}: Crowd-sourced emotional multimodal actors dataset",
  author    = "Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur,
               Ruben C and Nenkova, Ani and Verma, Ragini",
  abstract  = "People convey their emotional state in their face and voice. We
               present an audio-visual data set uniquely suited for the study
               of multi-modal emotion expression and perception. The data set
               consists of facial and vocal emotional expressions in sentences
               spoken in a range of basic emotional states (happy, sad, anger,
               fear, disgust, and neutral). 7,442 clips of 91 actors with
               diverse ethnic backgrounds were rated by multiple raters in
               three modalities: audio, visual, and audio-visual. Categorical
               emotion labels and real-value intensity values for the perceived
               emotion were collected using crowd-sourcing from 2,443 raters.
               The human recognition of intended emotion for the audio-only,
               visual-only, and audio-visual data are 40.9\%, 58.2\% and 63.6\%
               respectively. Recognition rates are highest for neutral,
               followed by happy, anger, disgust, fear, and sad. Average
               intensity levels of emotion are rated highest for visual-only
               perception. The accurate recognition of disgust and fear
               requires simultaneous audio-visual cues, while anger and
               happiness can be well recognized based on evidence from a single
               modality. The large dataset we introduce can be used to probe
               other questions concerning the audio-visual perception of
               emotion.",
  journal   = "IEEE Trans. Affect. Comput.",
  publisher = "Institute of Electrical and Electronics Engineers (IEEE)",
  volume    =  5,
  number    =  4,
  pages     = "377--390",
  month     =  oct,
  year      =  2014,
  keywords  = "Emotional corpora; facial expression; multi-modal recognition;
               voice expression",
  copyright = "https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html",
  language  = "en"
}

        """,
        descriptive_stats={
            "n_samples": {"train": 7440}
        },
    )

    # Override default column name in the subclass
    audio_column_name: str = "audio"
    label_column_name: str = "major_emotion"

    def get_candidate_labels(self) -> list[str]:
        unique_emotions = set(example[self.label_column_name] for example in self.dataset["train"])
    
        return [
            f"a phrase spoken with emotion {emotion}"
            for emotion in unique_emotions
        ]
