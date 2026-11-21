from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CQADupstackWebmastersNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWebmasters-NL",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a "
        "Dutch-translated version.",
        reference="https://huggingface.co/datasets/clips/beir-nl-cqadupstack",
        dataset={
            "path": "mteb/CQADupstackWebmasters-NL",
            "revision": "12d0619ff07bb8b825897af0dabee23ada29cd23",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date
        domains=["Written", "Non-fiction"],
        task_subtypes=["Duplicate Detection"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@inproceedings{banar2024beirnlzeroshotinformationretrieval,
  address = {Abu Dhabi, UAE},
  archiveprefix = {arXiv},
  author = {Lotfi, Ehsan and Banar, Nikolay and Daelemans, Walter},
  booktitle = {Proceedings of the 18th Workshop on Building and Using Comparable Corpora ({BUCC})},
  editor = {Sharoff, Serge and Terryn, Ayla Rigouts and Zweigenbaum, Pierre and Rapp, Reinhard},
  eprint = {2412.08329},
  month = {jan},
  pages = {36--45},
  publisher = {Association for Computational Linguistics},
  title = {{BEIR}-{NL}: Zero-shot Information Retrieval Benchmark for the {D}utch Language},
  url = {https://aclanthology.org/2025.bucc-1.5/},
  year = {2025},
}
""",
        adapted_from=["CQADupstackWebmastersRetrieval"],
    )
