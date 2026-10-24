from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class BirdCLEFSpeciesClustering(AbsTaskClustering):
    label_column_name: str = "primary_label"
    input_column_name: str = "recording"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="BirdCLEFSpeciesClustering",
        description=(
            "Audio clustering of wildlife recordings from the BirdCLEF+ 2025 dataset "
            "by species label. 1,000 recordings covering 50 sound-producing species "
            "(birds, amphibians, mammals and insects) from the Middle Magdalena Valley "
            "of Colombia, with exactly 20 samples per species."
        ),
        reference="https://huggingface.co/datasets/mteb/birdclef25-mini",
        dataset={
            "path": "mteb/birdclef25-mini",
            "revision": "582215665b247604b555da7ff4e071f82d3617db",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2025-01-01", "2025-12-31"),
        domains=["Spoken", "Speech", "Bioacoustics"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
