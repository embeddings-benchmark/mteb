from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HunSum2AbstractiveRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HunSum2AbstractiveRetrieval",
        dataset={
            "path": "mteb/HunSum2AbstractiveRetrieval",
            "revision": "039db2353a8d399d7ff1019d7c06d6f4e23857c1",
        },
        description=(
            "HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata."
        ),
        reference="https://aclanthology.org/2024.lrec-main.662/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hun-Latn"],
        main_score="ndcg_at_1",
        date=(
            "1848-12-15",
            "2024-03-19",
        ),
        domains=["News", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{barta2024news,
  address = {Torino, Italia},
  archiveprefix = {arXiv},
  author = {Barta, Botond and Lakatos, Dorina and Nagy, Attila and Nyist, Mil{\'a}n Konor and {\'A}cs, Judit},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation ({LREC}-{COLING} 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  eprint = {2404.03555},
  month = {may},
  pages = {7503--7509},
  publisher = {ELRA and ICCL},
  title = {From News to Summaries: Building a {H}ungarian Corpus for Extractive and Abstractive Summarization},
  url = {https://aclanthology.org/2024.lrec-main.662/},
  year = {2024},
}
""",
    )
