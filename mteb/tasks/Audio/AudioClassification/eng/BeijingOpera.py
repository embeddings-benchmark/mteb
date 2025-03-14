from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata
import datasets


class BeijingOpera(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="BeijingOpera",
        description="Instrument Source Classification. This dataset is a collection of audio examples of individual strokes spanning the four percussion instrument classes used in Beijing Opera.  There are four Beijing Opera instrument classes: Bangu (Clapper-drum), Naobo (Cymbals), Daluo (Large gong) and Xiaoluo (Small gong).",
        reference="https://huggingface.co/datasets/DynamicSuperb/InstrumentClassification_BeijingOperaInstrument",
        dataset={
            "path": "DynamicSuperb/InstrumentClassification_BeijingOperaInstrument",
            "revision": "d7279666447659c1adddb495aae0028c38219a32",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-03-06", "2025-03-06"),
        domains=["Music"],
        task_subtypes=["Instrument Source Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@misc{engel2017neuralaudiosynthesismusical,
            title={Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders}, 
            author={Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
            year={2017},
            eprint={1704.01279},
            archivePrefix={arXiv},
            primaryClass={cs.LG},
            url={https://arxiv.org/abs/1704.01279}, 
        }""",
        descriptive_stats={
            "n_samples": {"train": 289205, "validation": 12678, "test": 4096},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 50


    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=lang, **self.metadata_dict["dataset"]
            )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        print(self.dataset)

        for lang in self.hf_subsets:

            def extract_unique_labels(annotations):
                # Get unique labels from the annotations list
                labels = set()
                for annotation in annotations:
                    labels.add(annotation["label"])
                return list(labels)

            # Create new column with unique labels for each data point
            for split in self.dataset[lang]:
                self.dataset[lang][split] = self.dataset[lang][split].add_column(
                    "labels",
                    [
                        extract_unique_labels(annotations)
                        for annotations in self.dataset[lang][split]["label"]
                    ],
                )