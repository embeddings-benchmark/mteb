from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BirdSetMultilabelClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="BirdSet",
        description="BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics",
        reference="https://huggingface.co/datasets/DBD-research-group/BirdSet",
        dataset={
            "path": "DBD-research-group/BirdSet",
            "name": "HSN",
            "revision": "b0c14a03571a7d73d56b12c4b1db81952c4f7e64",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test_5s", "test"],
        eval_langs=[
            "eng-Latn",
        ],
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),  # Competition year
        domains=["Spoken", "Speech", "Bioacoustics"],
        task_subtypes=["Species Classification"],
        license="cc-by-nc-4.0",
        dialect=[],
        modalities=["audio"],
        bibtex_citation="""@misc{rauch2024birdsetlargescaledatasetaudio,
              title={BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics}, 
              author={Lukas Rauch and Raphael Schwinger and Moritz Wirth and René Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
              year={2024},
              eprint={2403.10380},
              archivePrefix={arXiv},
              primaryClass={cs.SD},
              url={https://arxiv.org/abs/2403.10380}, 
        }""",
        descriptive_stats={
            "n_samples": {"test_5s": 12000},  # 50 classes × 20 samples each
            "n_classes": 50,
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 21

    def dataset_transform(self):
        """Rename ebird_code_multilabel → labels and turn IDs → bird names."""
        if "ebird_code_multilabel" in self.dataset.column_names[self.eval_splits[0]]:
            self.dataset = self.dataset.rename_column(
                original_column_name="ebird_code_multilabel",
                new_column_name=self.label_column_name,  # "labels"
            )

        id2name = {
            0: "Gray-crowned Rosy-Finch",
            1: "White-crowned Sparrow",
            2: "American Pipit",
            3: "Spotted Sandpiper",
            4: "Rock Wren",
            5: "Brewer's Blackbird",
            6: "Dark-eyed Junco",
            7: "Fox Sparrow",
            8: "Clark's Nutcracker",
            9: "Mountain Bluebird",
            10: "Cassin's Finch",
            11: "Mallard",
            12: "Hermit Thrush",
            13: "American Robin",
            14: "Yellow-rumped Warbler",
            15: "Yellow Warbler",
            16: "Dusky Flycatcher",
            17: "Mountain Chickadee",
            18: "Orange-crowned Warbler",
            19: "Warbling Vireo",
            20: "Northern Flicker"
        }

        def ids_to_names(example):
            """Convert a list of ints to a list of bird-name strings."""
            return {
                self.label_column_name: [id2name[i] for i in example[self.label_column_name]]
            }

        # Apply to every split you evaluate on (e.g. "test_5s" and "test")
        self.dataset = self.dataset.map(
            ids_to_names,
            num_proc=4,           # speed-up; adjust or drop if you’re low on RAM
        )

        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=self.eval_splits,
            label=self.label_column_name,
            n_samples=2048,
        )
