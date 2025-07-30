from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioSetMultilingualClassification(AbsTaskAudioMultilabelClassification):
    superseded_by = "AudioSetMini"
    metadata = TaskMetadata(
        name="AudioSet",
        description="AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos.",
        reference="https://huggingface.co/datasets/agkphysics/AudioSet",
        dataset={
            "path": "agkphysics/AudioSet",
            "revision": "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="lrap",
        date=(
            "2016-01-01",
            "2017-01-30",
        ),
        domains=["Web", "Music", "Speech", "Scene"],
        task_subtypes=[
            "Environment Sound Classification",
            "Music Instrument Recognition",
            "Vocal Sound Classification",
            "Gunshot Audio Classification",
        ],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{45857,
  address = {New Orleans, LA},
  author = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
  booktitle = {Proc. IEEE ICASSP 2017},
  title = {Audio Set: An ontology and human-labeled dataset for audio events},
  year = {2017},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "human_labels"


# Sampled using scripts/data/audioset/create_data.ipynb
class AudioSetMiniMultilingualClassification(AbsTaskAudioMultilabelClassification):
    metadata = TaskMetadata(
        name="AudioSetMini",
        description="AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos. This is a mini version that is sampled from the original dataset.",
        reference="https://huggingface.co/datasets/agkphysics/AudioSet",
        dataset={
            "path": "mteb/audioset",
            "revision": "168a7e681ee40609129535d49855c7e3e77e5efa",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="lrap",
        date=(
            "2016-01-01",
            "2017-01-30",
        ),
        domains=["Web", "Music", "Speech", "Scene"],
        task_subtypes=[
            "Environment Sound Classification",
            "Music Instrument Recognition",
            "Vocal Sound Classification",
            "Gunshot Audio Classification",
        ],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{45857,
  address = {New Orleans, LA},
  author = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
  booktitle = {Proc. IEEE ICASSP 2017},
  title = {Audio Set: An ontology and human-labeled dataset for audio events},
  year = {2017},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "human_labels"
