from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class MedPTClustering(AbsTaskClustering):
    """MedPT medical-specialty clustering. 12 specialties × ~50 questions."""

    metadata = TaskMetadata(
        name="MedPTClustering",
        description=(
            "Cluster Brazilian Portuguese medical questions by specialty. Uses "
            "the 12 largest specialties from AKCIT/MedPT (e.g., Cardiologista, "
            "Ortopedista, Ginecologista, etc.), 50 questions sampled per "
            "specialty (~600 total). Replaces CSTNewsClustering which degenerated "
            "to v_measure=1.0 for every model. Tests embedding discrimination "
            "across PT-BR medical vocabulary."
        ),
        reference="https://huggingface.co/datasets/AKCIT/MedPT",
        dataset={
            "path": "MTEB-BR/MedPTClustering",
            "revision": "230d08b2cdf8fe8447fc65b2aa8aa6b13b1776dc",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{farber-etal-2026-medpt,
  title = {{M}ed{PT}: A Massive Medical Question Answering Dataset for {B}razilian-{P}ortuguese Speakers},
  author = {Farber, Fernanda Bufon and Brito, Iago Alves and Dollis, Julia Soares and Ribeiro, Pedro Schindler Freire Brasil and Sousa, Rafael Teixeira and Galv{\~a}o Filho, Arlindo Rodrigues},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = may,
  year = {2026},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  pages = {2419--2429},
  doi = {10.63317/5mrjjtbsf6rv},
  url = {https://arxiv.org/abs/2511.11878},
}
""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
