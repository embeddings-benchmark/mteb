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
        license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{piczak2015dataset,
        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
        author = {Piczak, Karol J.},
        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
        date = {2015-10-13},
        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
        doi = {10.1145/2733373.2806390},
        location = {{Brisbane, Australia}},
        isbn = {978-1-4503-3459-4},
        publisher = {{ACM Press}},
        pages = {1015--1018}
    }""",
        descriptive_stats={
            "n_samples": {"train": 2000},
        },
    )

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset["train"])

        df = df.rename(columns={"target": "label"})
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

        all_files = [audio["array"].tolist() for audio in df["audio"]]
        all_labels = df["label"].values.tolist()

        print("done!")

        print("Generating dissimilar pairs: ")
        num_similar = len(similar_pairs)
        while len(dissimilar_pairs) < num_similar:
            idx1, idx2 = random.sample(range(len(all_files)), 2)
            if all_labels[idx1] != all_labels[idx2]:
                dissimilar_pairs.append([all_files[idx1], all_files[idx2], [0]])

        pairs = similar_pairs + dissimilar_pairs
        random.shuffle(pairs)
        print("done!")

        print(f"Number of pairs: {len(pairs)}")
        audio1, audio2, label = zip(*pairs)

        audio1 = list(audio1)
        audio2 = list(audio2)
        label = list(label)

        HF_ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        print("Zipping features and generating dataset...")
        # convert back to HF dataset
        self.dataset = datasets.DatasetDict({"test": HF_ds})
        print("done!")
