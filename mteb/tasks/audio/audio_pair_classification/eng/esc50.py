from __future__ import annotations

import logging

import datasets
import numpy as np
from tqdm import tqdm

from mteb.abstasks.audio.abs_task_audio_pair_classification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class ESC50PairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="ESC50PairClassification",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "ashraq/esc50",
            "revision": "e3e2a63ffff66b9a9735524551e3818e96af03ee",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-01-07", "2023-01-07"),
        domains=["Encyclopaedic"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
    )

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        ds = self.dataset["train"]
        logger.info(f"Starting dataset transformation with seed {self.seed}...")

        ds = ds.rename_column("target", "label")

        # group indices by label
        label2indices = {}
        for idx, label in enumerate(ds["label"]):
            label2indices.setdefault(label, []).append(idx)

        rng = np.random.default_rng(self.seed)
        similar_pairs = []

        for label, indices in tqdm(label2indices.items()):
            indices = np.array(indices)
            rng.shuffle(indices)
            # create pairs, handling odd number of samples
            for i in range(0, len(indices) - 1, 2):
                idx1, idx2 = indices[i], indices[i + 1]
                similar_pairs.append((int(idx1), int(idx2), 1))

        num_similar = len(similar_pairs)
        logger.info(f"Number of similar pairs: {num_similar}")

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

        logger.info(f"Number of dissimilar pairs: {len(dissimilar_pairs)}")

        pairs = similar_pairs + dissimilar_pairs
        rng.shuffle(pairs)

        audio1 = [ds[idx1]["audio"]["array"] for idx1, idx2, _ in pairs]
        audio2 = [ds[idx2]["audio"]["array"] for idx1, idx2, _ in pairs]
        label = [[lbl] for _, _, lbl in pairs]

        ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        self.dataset = datasets.DatasetDict({"test": ds})
