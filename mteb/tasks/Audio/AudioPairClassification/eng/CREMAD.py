from __future__ import annotations

import random

import datasets
import pandas as pd

from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

random.seed(42)


class CREMADPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="CREMADPairClassification",
        description="Classifying pairs as having same or different emotions in actor's voice recordings of text spoken in 6 different emotions",
        reference="https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/",
        dataset={
            "path": "AbstractTTS/CREMA-D",
            "revision": "5a172780c79bbddb9b326a2c830447c550a216a4",
        },
        type="AudioPairClassification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        modalities=["audio"],
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
        descriptive_stats={"n_samples": {"train": 7440}},
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset["train"])

        df = df.rename(columns={"major_emotion": "label"})
        label_id = {label: i for i, label in enumerate(df["label"].unique())}
        df["label"] = [label_id[label] for label in df["label"]]
        grouped = [df.loc[df["label"] == label] for label in df["label"].unique()]

        similar_pairs = []
        dissimilar_pairs = []

        print("Generating similar pairs: ")
        for group in grouped:
            files = [audio["array"].tolist() for audio in group["audio"]]
            random.shuffle(files)
            similar_pairs.extend(
                [[files[i], files[i + 1], [1]] for i in range(0, len(files) - 1, 2)]
            )
        print("done!")

        all_files = [audio["array"].tolist() for audio in df["audio"]]
        all_labels = df["label"].values.tolist()

        print("Generating dissimilar pairs: ")
        num_similar = len(similar_pairs)
        while len(dissimilar_pairs) < num_similar:
            idx1, idx2 = random.sample(range(len(all_files)), 2)
            if all_labels[idx1] != all_labels[idx2]:
                dissimilar_pairs.append([all_files[idx1], all_files[idx2], [0]])
        print("done!")

        pairs = similar_pairs + dissimilar_pairs
        random.shuffle(pairs)

        print(f"Number of pairs: {len(pairs)}")

        audio1, audio2, label = zip(*pairs)
        audio1 = list(audio1)
        audio2 = list(audio2)
        label = list(label)

        HF_ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        print("Zipping features and generating dataset: ")
        # convert back to HF dataset
        self.dataset = datasets.DatasetDict({"test": HF_ds})
        print("done!")
