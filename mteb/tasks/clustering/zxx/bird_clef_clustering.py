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
@inproceedings{canas_overview_2025,
  author = {Ca{\~n}as, Juan Sebasti{\'a}n and Kahl, Stefan and Denton, Tom and Toro-G{\'o}mez, Mar{\'i}a Paula and Rodriguez-Buritica, Susana and Benavides-Lopez, Jose Luis and Ulloa, Juan Sebasti{\'a}n and Caycedo-Rosales, Paula and Klinck, Holger and Go{\"e}au, Herv{\'e} and Planqu{\'e}, Robert and Vellinga, Willem-Pier and Joly, Alexis},
  title = {Overview of {BirdCLEF+} 2025: Multi-taxonomic sound identification in the {Middle Magdalena}, {Colombia}},
  booktitle = {{CEUR} Workshop},
  editor = {Faggioli, Guglielmo and Ferro, Nicola and Rosso, Paolo and Spina, Damiano},
  series = {{CEUR} Workshop Proceedings},
  volume = {4038},
  pages = {2909--2919},
  publisher = {CEUR},
  address = {Madrid, Spain},
  year = {2025},
  month = sep,
  url = {https://hal.inrae.fr/hal-05319248},
}
""",
    )
