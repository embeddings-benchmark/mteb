from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSD50HFMultilingualClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="FSD50K",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/Chand0320/fsd50k_hf",
        dataset={
            "path": "Chand0320/fsd50k_hf",
            "revision": "ca72d33100074e2933437e844028c941d8e8f065",
        },  # this is actually used to download the data
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-01-30",
        ),  # Estimated date when this dataset was committed, what should be the second tuple?
        domains=["Web"],  # obtained from Freesound - online collaborative platform
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@ARTICLE{9645159,
                    author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
                    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
                    title={FSD50K: An Open Dataset of Human-Labeled Sound Events}, 
                    year={2022},
                    volume={30},
                    number={},
                    pages={829-852},
                    keywords={Videos;Task analysis;Labeling;Vocabulary;Speech recognition;Ontologies;Benchmark testing;Audio dataset;sound event;recognition;classification;tagging;data collection;environmental sound},
                    doi={10.1109/TASLP.2021.3133208}}
        """,
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 8

    def dataset_transform(self):
        # labels column is a string of comma separated labels, this function converts it to a list of labels
        self.dataset = self.dataset.map(
            lambda x: {
                self.label_column_name: x[self.label_column_name].split(","),
            }
        )
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=self.eval_splits,
            label=self.label_column_name,
            n_samples=2048,
        )
