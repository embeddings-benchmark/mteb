from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GerDaLIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GerDaLIR",
        description="GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "mteb/GerDaLIR",
            "revision": "735f2cca1298426b3c792de3527b4bdc18537fc0",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal"],
        task_subtypes=[],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{wrzalik-krechel-2021-gerdalir,
  address = {Punta Cana, Dominican Republic},
  author = {Wrzalik, Marco  and
Krechel, Dirk},
  booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
  month = nov,
  pages = {123--128},
  publisher = {Association for Computational Linguistics},
  title = {{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval},
  url = {https://aclanthology.org/2021.nllp-1.13},
  year = {2021},
}
""",
    )
