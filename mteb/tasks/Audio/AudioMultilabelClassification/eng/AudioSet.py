from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioSetMultilingualClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="AudioSet",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/agkphysics/AudioSet",
        dataset={
            "path": "agkphysics/AudioSet",
            "revision": "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef",
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
        task_subtypes=[
            "Environment Sound Classification"
        ],  # Since this dataset has sounds of ALL types, this seems to be the best option
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{45857,title	= {Audio Set: An ontology and human-labeled dataset for audio events},author	= {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},year	= {2017},booktitle	= {Proc. IEEE ICASSP 2017},address	= {New Orleans, LA}}
        """,
        descriptive_stats={
            "n_samples": {"test": 8961},  # Need to change
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 8  # Need to change
