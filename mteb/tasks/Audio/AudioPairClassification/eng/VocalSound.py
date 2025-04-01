from __future__ import annotations

import random

import datasets
import pandas as pd
from tqdm import tqdm

from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

random.seed(42)


class VocalSoundPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="VocalSoundPairClassification",
        description="Recognizing whether two audio clips are the same human vocal expression (laughing, sighing, etc.)",
        reference="https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations",
        dataset={
            "path": "DynamicSuperb/VocalSoundRecognition_VocalSound",
            "revision": "beb7fe456e01f1a9959daae8dd507fa7790f4b62",
        },
        type="AudioPairClassification",
        category="t2t",  # no audio category yet
        eval_splits=["test"],
        eval_langs=["eng-latn"],
        main_score="max_ap",
        domains=["Spoken"],  # no task domain existing for music, probably should add
        task_subtypes=["Emotion classification"],
        license="not specified",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{inproceedings,
author = {Gong, Yuan and Yu, Jin and Glass, James},
year = {2022},
month = {05},
pages = {151-155},
title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
doi = {10.1109/ICASSP43922.2022.9746828}
}
        """,
        descriptive_stats={"n_samples": {"train": 6400}},
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset["test"])
        df["label"] = pd.factorize(df["label"])[0]
        grouped = [df.loc[df["label"] == label] for label in df["label"].unique()]

        similar_pairs = []
        dissimilar_pairs = []

        print("Generating similar pairs: ")
        for group in tqdm(grouped):
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
        print("Number of pairs: ", len(pairs))
        random.shuffle(pairs)

        audio1, audio2, label = zip(*pairs)

        audio1 = list(audio1)
        audio2 = list(audio2)
        label = list(label)

        HF_ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        print("Generating dataset: ")
        # convert back to HF dataset
        self.dataset = datasets.DatasetDict({"test": HF_ds})

        print("done!")
