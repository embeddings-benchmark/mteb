from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class VehicleSoundClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="VehicleSoundClustering",
        description="Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class))",
        reference="https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset",
        dataset={
            "path": "mteb/Vehicle_sounds_classification_dataset",
            "revision": "5f905f9544d0c78b607639662c88c2ddd1216384",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
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
@inproceedings{bazilinskyy_auditory_2018,
  title = {An auditory dataset of passing vehicles recorded with a smartphone},
  keywords = {traffic, microphone, smartphone, sound analysis, sound processing},
  author = {Pavlo Bazilinskyy and \{van der Aa\}, Arne and Michael Schoustra and John Spruit and Laurens Staats and \{van der Vlist\}, \{Klaas Jan\} and \{de Winter\}, Joost},
  year = {2018},
  language = {English},
  isbn = {978-94-6186-910-4},
  pages = {417--422},
  editor = {Imre Horv{\'a}th and Su{\'a}rez, \{Jos{\'e} Pablo\}},
  booktitle = {Proceedings of the 12th International Symposium on Tools and Methods of Competitive Engineering (TMCE 2018)},
  publisher = {Delft University of Technology},
  address = {Netherlands},
}
""",
    )

    label_column_name: str = "label"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
