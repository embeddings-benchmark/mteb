from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GerDaLIR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GerDaLIR",
        description="GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "mteb/GerDaLIRSmall",
            "revision": "b199f38071bc06a2cb86c4c10d57ecee6c46056a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
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
  abstract = {We present GerDaLIR, a German Dataset for Legal Information Retrieval based on case documents from the open legal information platform Open Legal Data. The dataset consists of 123K queries, each labelled with at least one relevant document in a collection of 131K case documents. We conduct several baseline experiments including BM25 and a state-of-the-art neural re-ranker. With our dataset, we aim to provide a standardized benchmark for German LIR and promote open research in this area. Beyond that, our dataset comprises sufficient training data to be used as a downstream task for German or multilingual language models.},
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
