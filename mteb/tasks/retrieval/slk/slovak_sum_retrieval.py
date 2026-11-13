from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakSumRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakSumRetrieval",
        description="SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand news articles with titles and short abstracts obtained from multiple Slovak newspapers. Originally intended as a summarization task, but since no human annotations were provided here reformulated to a retrieval task.",
        reference="https://huggingface.co/datasets/NaiveNeuron/slovaksum",
        dataset={
            "path": "mteb/SlovakSumRetrieval",
            "revision": "bc98bfcde92b79fd183f98c6ed3b7ebd87a74eed",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2015-04-26", "2022-01-11"),
        domains=["News", "Social", "Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="openrail",
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
    )
