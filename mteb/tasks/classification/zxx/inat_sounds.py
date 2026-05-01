from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class INatSoundsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="INatSounds",
        description="iNaturalist Sounds dataset for species identification from audio. Contains recordings of 5,500+ species across birds, insects, amphibians, mammals, and reptiles, contributed by 27,000+ citizen scientists. Test split contains 49,527 recordings from 2023 observations.",
        reference="https://github.com/visipedia/inat_sounds",
        dataset={
            "path": "mteb/inat_sounds",
            "revision": "6db625bf80d71ded64690022f0df56ceae972b6e",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
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
        bibtex_citation=r"""
@article{chasmai2025inatsounds,
  archiveprefix = {arXiv},
  author = {Mustafa Chasmai and Alexander Shepard and Subhransu Maji and Grant Van Horn},
  eprint = {2506.00343},
  primaryclass = {cs.SD},
  title = {The iNaturalist Sounds Dataset},
  url = {https://arxiv.org/abs/2506.00343},
  year = {2025},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    n_experiments = 1
    is_cross_validation: bool = True
    n_splits = 3
