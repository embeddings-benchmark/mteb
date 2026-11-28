from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ClimateFEVERNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER-NL",
        description="CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world "
        "claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-climate-fever",
        dataset={
            "path": "clips/beir-nl-climate-fever",
            "revision": "8d4f81ded229ee2bfd63b2d30a9df5d678abb1f7",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2020-12-01", "2020-12-01"),  # best guess: based on publication date
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
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
        adapted_from=["ClimateFEVER"],
    )
