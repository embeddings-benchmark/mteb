from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NQPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-PL",
        description="Natural Questions: A Benchmark for Question Answering Research",
        reference="https://ai.google.com/research/NaturalQuestions/",
        dataset={
            "path": "mteb/NQ-PL",
            "revision": "b784c0399830a24e4ce0f9df2bb5b6be7fc8b246",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2023-05-01", "2024-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
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
        adapted_from=["NQ"],
    )


class NQPLHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-PLHardNegatives",
        description="Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        dataset={
            "path": "mteb/NQ-PLHardNegatives",
            "revision": "cd0cab9433aa7f4e939883d822fb93e362fd493a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),  # original publication year
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",  # original license
        annotations_creators="human-annotated",  # original annotation method
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
        adapted_from=["NQ"],
    )
