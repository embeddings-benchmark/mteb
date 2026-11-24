from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TRECCOVIDPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-PL",
        description="TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        dataset={
            "path": "mteb/TRECCOVID-PL",
            "revision": "1e710582482d4199ff690e5dd2491a70627523f5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2019-12-01",
            "2022-12-31",
        ),  # approximate date of covid pandemic start and end (best guess)
        domains=["Academic", "Medical", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
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
        adapted_from=["TRECCOVID"],
    )
