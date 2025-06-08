from __future__ import annotations

import logging

import datasets
import numpy as np
import tqdm

from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class VoxPopuliAccentPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="VoxPopuliAccentPairClassification",
        description="Classifying same or different regional accent of English",
        reference="https://aclanthology.org/2021.acl-long.80/",
        dataset={
            "path": "facebook/voxpopuli",
            "name": "en_accented",
            "revision": "719aaef8225945c0d80b277de6c79aa42ab053d5",
        },
        type="AudioPairClassification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-latn"],
        main_score="max_ap",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc0-1.0",
        modalities=["audio"],
        sample_creation="created",
        descriptive_stats={"n_samples": {"train": 4200}},
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        ds = self.dataset["test"]
        logger.info(f"Starting dataset transformation with seed {self.seed}...")

        ds = ds.rename_column("accent", "label")

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
        for label, indices in tqdm.tqdm(label2indices.items()):
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

        for label, indices in tqdm.tqdm(label2indices.items()):
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
