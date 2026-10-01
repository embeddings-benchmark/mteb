from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BirdCLEFClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BirdCLEF",
        description="BirdCLEF+ 2025 dataset for species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia. Downsampled to 50 classes with 20 samples each.",
        reference="https://huggingface.co/datasets/christopher/birdclef-2025",
        dataset={
            "path": "mteb/birdclef25-mini",
            "revision": "582215665b247604b555da7ff4e071f82d3617db",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),  # Competition year
        domains=["Spoken", "Speech", "Bioacoustics"],
        task_subtypes=["Species Classification"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{canas2025overview,
  author = {Ca{\~n}as, Juan Sebasti{\'a}n and Kahl, Stefan and Denton, Tom and Toro-G{\'o}mez, Maria Paula and Rodriguez-Buritica, Susana and Benavides-Lopez, Jose Luis and Ulloa, Juan Sebasti{\'a}n and Caycedo-Rosales, Paula and Klinck, Holger and Go{\"e}au, Herv{\'e} and others},
  booktitle = {Conference and Labs of the Evaluation Forum (CLEF 2025)},
  number = {4038},
  organization = {CEUR-WS},
  pages = {2909--2919},
  title = {Overview of BirdCLEF+ 2025: Multi-taxonomic sound identification in the Middle Magdalena, Colombia},
  year = {2025},
}
""",
    )

    input_column_name: str = "recording"
    label_column_name: str = "primary_label"

    is_cross_validation: bool = True
