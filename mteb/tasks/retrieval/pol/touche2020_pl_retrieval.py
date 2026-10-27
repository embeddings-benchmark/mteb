from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Touche2020PL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020-PL",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "mteb/Touche2020-PL",
            "revision": "ec535fbc3776cfc96d72ad0cb5e8f81e74b2fd4e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2020-09-23",
            "2020-09-23",
        ),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{wojtasik2024beirpl,
  address = {Torino, Italia},
  archiveprefix = {arXiv},
  author = {Wojtasik, Konrad and Wo{\l}owiec, Kacper and Shishkin, Vadim and Janz, Arkadiusz and Piasecki, Maciej},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation ({LREC}-{COLING} 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  eprint = {2305.19840},
  month = {may},
  pages = {2149--2160},
  publisher = {ELRA and ICCL},
  title = {{BEIR}-{PL}: Zero Shot Information Retrieval Benchmark for the {P}olish Language},
  url = {https://aclanthology.org/2024.lrec-main.194/},
  year = {2024},
}
""",
        adapted_from=["Touche2020"],
    )
