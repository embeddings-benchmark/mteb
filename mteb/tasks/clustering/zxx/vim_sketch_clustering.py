from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class VimSketchImitationClustering(AbsTaskClustering):
    label_column_name: str = "label"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="VimSketchImitationClustering",
        description=(
            "Clustering of vocal imitations from the VimSketch dataset by the "
            "sound they imitate: 1,230 human vocal imitations covering 50 "
            "reference-sound classes sampled with a fixed seed (at most 40 "
            "imitations per class). Labels are the imitated reference classes."
        ),
        reference="https://zenodo.org/record/2596911",
        dataset={
            "path": "dukesun99/VimSketch",
            "name": "clustering",
            "revision": "466e0ea0ed8f50bad9c240f3bfc8426c08430aa2",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2015-01-01", "2019-03-18"),
        domains=["AudioScene", "Spoken"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{cartwright_vocalsketch_2015,
  author = {Cartwright, Mark and Pardo, Bryan},
  title = {VocalSketch: Vocally Imitating Audio Concepts},
  year = {2015},
  isbn = {9781450331456},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/2702123.2702387},
  doi = {10.1145/2702123.2702387},
  booktitle = {Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems},
  pages = {43–46},
  numpages = {4},
  keywords = {vocal imitation, user interaction, data set, audio software},
  location = {Seoul, Republic of Korea},
  series = {CHI '15},
}

@inproceedings{kim_vocal_2018,
  author = {Kim, Bongjun and Ghei, Madhav and Pardo, Bryan and Duan, Zhiyao},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
  pages = {148--152},
  title = {Vocal Imitation Set: a dataset of vocally imitated sound events using the AudioSet ontology},
  year = {2018},
}
""",
    )
