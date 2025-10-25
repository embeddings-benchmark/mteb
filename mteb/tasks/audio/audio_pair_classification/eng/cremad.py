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
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-01-01", "2014-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="http://opendatacommons.org/licenses/odbl/1.0/",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{Cao2014-ih,
  abstract = {People convey their emotional state in their face and voice. We
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
emotion.},
  author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur,
Ruben C and Nenkova, Ani and Verma, Ragini},
  copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
  journal = {IEEE Trans. Affect. Comput.},
  keywords = {Emotional corpora; facial expression; multi-modal recognition;
voice expression},
  language = {en},
  month = oct,
  number = {4},
  pages = {377--390},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  title = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
  volume = {5},
  year = {2014},
}
""",
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        ds = self.dataset["train"]
        logger.info(f"Starting dataset transformation with seed {self.seed}...")

        ds = ds.rename_column("major_emotion", "label")

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

        for label, indices in tqdm(label2indices.items()):
            indices = np.array(indices)
            rng.shuffle(indices)
            # create pairs, handling odd number of samples
            for i in range(0, len(indices) - 1, 2):
                idx1, idx2 = indices[i], indices[i + 1]
                similar_pairs.append((int(idx1), int(idx2), 1))

        num_similar = len(similar_pairs)
        logger.info(f"Found similar pairs: {num_similar}")

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

        logger.info(
            f"Using {len(dissimilar_pairs)} dissimilar pairs, {len(similar_pairs)} similar pairs"
        )

        pairs = similar_pairs + dissimilar_pairs
        rng.shuffle(pairs)

        audio1 = [ds[idx1]["audio"]["array"] for idx1, idx2, _ in pairs]
        audio2 = [ds[idx2]["audio"]["array"] for idx1, idx2, _ in pairs]
        label = [[lbl] for _, _, lbl in pairs]

        ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        self.dataset = datasets.DatasetDict({"test": ds})
