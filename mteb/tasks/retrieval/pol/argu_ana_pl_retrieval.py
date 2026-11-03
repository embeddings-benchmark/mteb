from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ArguAnaPL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAna-PL",
        description="ArguAna-PL",
        reference="https://huggingface.co/datasets/clarin-knext/arguana-pl",
        dataset={
            "path": "mteb/ArguAna-PL",
            "revision": "0019de73979daf75f6526a0fc5ef77ebce27b8f9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2023-05-01", "2024-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-sa-4.0",
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
        adapted_from=["ArguAna"],
    )
