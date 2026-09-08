from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakSumURLClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="SlovakSumURLClustering",
        description="Clustering of Slovak news articles from SlovakSum dataset based on the URL structure. Articles are organized into 12 editorial categories including sports, culture, economy, health, travel, politics, and technology sections.",
        reference="https://aclanthology.org/2024.lrec-main.1298/",
        dataset={
            "path": "mteb/SlovakSumURLClustering",
            "revision": "3410778ab05bf1f002fa26eab148410191427dab",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2015-04-26", "2022-01-11"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ondrejova-suppa-2024-slovaksum,
  address = {Torino, Italia},
  author = {Ondrejová, Viktória and Šuppa, Marek},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  month = may,
  pages = {14916--14922},
  publisher = {ELRA and ICCL},
  title = {SlovakSum: A Large Scale Slovak Summarization Dataset},
  url = {https://aclanthology.org/2024.lrec-main.1298/},
  year = {2024},
}
""",
        prompt="Identify the topic or theme of the given text.",
    )
