from __future__ import annotations

import logging

import datasets
import numpy as np
from tqdm import tqdm

from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


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
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-sa-4.0",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{inproceedings,
  author = {Gong, Yuan and Yu, Jin and Glass, James},
  doi = {10.1109/ICASSP43922.2022.9746828},
  month = {05},
  pages = {151-155},
  title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
  year = {2022},
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
        ds = self.dataset["test"]
        logger.info(f"Starting dataset transformation with seed {self.seed}...")

        # convert string labels to int
        unique_labels = sorted(set(ds["label"]))
        label2int = {label: idx for idx, label in enumerate(unique_labels)}
        ds = ds.map(lambda x: {"label": label2int[x["label"]]})

        # group indices by label
        label2indices = {}
        for idx, label in enumerate(ds["label"]):
            label2indices.setdefault(label, []).append(idx)

        rng = np.random.default_rng(self.seed)
        similar_pairs = []

        logger.info("Generating similar pairs:")
        for label, indices in tqdm(label2indices.items()):
            indices = np.array(indices)
            rng.shuffle(indices)
            # create pairs, handling odd number of samples
            for i in range(0, len(indices) - 1, 2):
                idx1, idx2 = indices[i], indices[i + 1]
                similar_pairs.append((int(idx1), int(idx2), 1))

        num_similar = len(similar_pairs)
        logger.info(f"Found similar pairs: {num_similar}")

        logger.info("Generating dissimilar pairs:")
        labels = list(label2indices.keys())
        dissimilar_pairs = []

        # pre-compute the candidate indices for each label
        label_candidates = {}
        for label in labels:
            other_labels = [l for l in labels if l != label]
            label_candidates[label] = []
            for other_label in other_labels:
                label_candidates[label].extend(label2indices[other_label])

        for label, indices in tqdm(label2indices.items()):
            candidates = label_candidates[label]
            if not candidates:
                continue

            n_pairs = min(len(indices), num_similar // len(labels))

            sampled_indices = rng.choice(indices, size=n_pairs, replace=False)

            sampled_candidates = rng.choice(candidates, size=n_pairs, replace=True)

            for idx1, idx2 in zip(sampled_indices, sampled_candidates):
                dissimilar_pairs.append((int(idx1), int(idx2), 0))

        # ensure same number of similar and dissimilar pairs
        min_pairs = min(len(similar_pairs), len(dissimilar_pairs))
        similar_pairs = similar_pairs[:min_pairs]
        dissimilar_pairs = dissimilar_pairs[:min_pairs]

        logger.info(f"Using {len(dissimilar_pairs)} dissimilar pairs")
        logger.info(f"Using {len(similar_pairs)} similar pairs")

        pairs = similar_pairs + dissimilar_pairs
        rng.shuffle(pairs)

        audio1 = [ds[idx1]["audio"]["array"] for idx1, idx2, _ in pairs]
        audio2 = [ds[idx2]["audio"]["array"] for idx1, idx2, _ in pairs]
        label = [[lbl] for _, _, lbl in pairs]

        logger.info("Creating dataset...")
        HF_ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        logger.info("Generating final dataset...")
        self.dataset = datasets.DatasetDict({"test": HF_ds})
        logger.info("done!")
