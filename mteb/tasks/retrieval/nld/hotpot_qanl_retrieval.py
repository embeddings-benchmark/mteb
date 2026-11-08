from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HotpotQANL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-NL",
        dataset={
            "path": "clips/beir-nl-hotpotqa",
            "revision": "3ab9eaff3e1d6f7f81914ae3b2bdd1e8511cf9a8",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            "supervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is "
            "a Dutch translation. "
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
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
        adapted_from=["HotpotQA"],
    )
