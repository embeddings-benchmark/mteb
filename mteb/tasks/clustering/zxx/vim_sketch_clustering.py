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
@inproceedings{cartwright2015vocalsketch,
  author = {Cartwright, Mark and Pardo, Bryan},
  booktitle = {Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems},
  pages = {43--46},
  title = {VocalSketch: Vocally Imitating Audio Concepts},
  year = {2015},
}

@inproceedings{kim2018vocal,
  author = {Kim, Bongjun and Ghei, Madhav and Pardo, Bryan and Duan, Zhiyao},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
  pages = {148--152},
  title = {Vocal Imitation Set: a dataset of vocally imitated sound events using the AudioSet ontology},
  year = {2018},
}
""",
    )
