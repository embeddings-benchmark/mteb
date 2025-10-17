from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GerDaLIRSmall(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GerDaLIRSmall",
        description="The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "mteb/GerDaLIRSmall",
            "revision": "48327de6ee192e9610f3069789719788957c7abd",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
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
