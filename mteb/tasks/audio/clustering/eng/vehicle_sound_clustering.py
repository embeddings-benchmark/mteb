from __future__ import annotations

from mteb.abstasks.audio.abs_task_audio_clustering import AbsTaskAudioClustering
from mteb.abstasks.task_metadata import TaskMetadata


class VehicleSoundClustering(AbsTaskAudioClustering):
    metadata = TaskMetadata(
        name="VehicleSoundClustering",
        description="Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class))",
        reference="https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset",
        dataset={
            "path": "DynamicSuperb/Vehicle_sounds_classification_dataset",
            "revision": "9ad231d349f9d0bddbf20a83d0d7635dccbd2501",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-06-20", "2024-06-20"),
        domains=["Scene"],
        task_subtypes=["Vehicle Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{inproceedings,
  author = {Bazilinskyy, Pavlo and Aa, Arne and Schoustra, Michael and Spruit, John and Staats, Laurens and van der Vlist, Klaas Jan and de Winter, Joost},
  month = {05},
  pages = {},
  title = {An auditory dataset of passing vehicles recorded with a smartphone},
  year = {2018},
}
""",
    )

    label_column_name: str = "label"
    max_fraction_of_documents_to_embed = None
