from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class VoiceGenderClustering(AbsTaskClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="VoiceGenderClustering",
        description="Clustering audio recordings based on gender (male vs female).",
        reference="https://huggingface.co/datasets/mmn3690/voice-gender-clustering",
        dataset={
            "path": "mteb/VoiceGenderClustering",
            "revision": "def21cdda438562bf8515d1daa9b47ab6ea6dec6",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Gender Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Chung18b,
  author = {Joon Son Chung and Arsha Nagrani and Andrew Zisserman},
  booktitle = {Proceedings of Interspeech},
  title = {VoxCeleb2: Deep Speaker Recognition},
  year = {2018},
}
""",
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "audio"
