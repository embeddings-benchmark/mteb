from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@article{chasmai2025inatsounds,
  archiveprefix = {arXiv},
  author = {Mustafa Chasmai and Alexander Shepard and Subhransu Maji and Grant Van Horn},
  eprint = {2506.00343},
  primaryclass = {cs.SD},
  title = {The iNaturalist Sounds Dataset},
  url = {https://arxiv.org/abs/2506.00343},
  year = {2025},
}
"""


class INatSoundsMiniClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="INatSoundsMini",
        description="Stratified subsample (2,048 samples, 1,212 species) of the iNaturalist Sounds dataset for species identification from audio. Covers birds, insects, amphibians, mammals, and reptiles.",
        reference="https://github.com/visipedia/inat_sounds",
        dataset={
            "path": "mteb/INatSoundsMini",
            "revision": "63416d55da8fe9f7043e9396edaa7957cad7950c",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2024-01-01", "2025-06-01"),
        domains=["Spoken", "Bioacoustics"],
        task_subtypes=["Species Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    is_cross_validation: bool = True
