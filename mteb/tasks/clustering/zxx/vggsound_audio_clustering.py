from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{chen2020vggsound,
  author = {Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle = {ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech
               and Signal Processing (ICASSP)},
  doi = {10.1109/ICASSP40776.2020.9053174},
  organization = {IEEE},
  pages = {721-725},
  title = {VGGSound: A Large-Scale Audio-Visual Dataset},
  year = {2020},
}
"""


class VGGSoundAudioClustering(AbsTaskClustering):
    label_column_name: str = "label"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="VGGSoundAudioClustering",
        description=(
            "Audio-only clustering of VGGSound: 9,888 YouTube audio clips spanning "
            "309 sound classes (capped at 32 per class), covering a diverse range of "
            "sounds including speech, music, environmental sounds and animal calls. "
            "Evaluates whether audio embeddings can cluster semantically related sounds."
        ),
        reference="https://arxiv.org/abs/2004.14368",
        dataset={
            "path": "mteb/VGGSound",
            "revision": "a994211ca0996558ff6cbd6977b4c1749f49c889",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2020-05-04", "2020-05-08"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )
