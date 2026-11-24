from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class QuoraNLRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="Quora-NL",
        dataset={
            "path": "clips/beir-nl-quora",
            "revision": "698f8a5922a49e50e3c71b52a2e9d63b22756843",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            " question, find other (duplicate) questions. QuoraRetrieval-NL is a Dutch translation."
        ),
        reference="https://huggingface.co/datasets/clips/beir-nl-quora",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2016-03-01", "2016-03-01"),
        domains=["Written"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
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
        adapted_from=["QuoraRetrieval"],
    )
