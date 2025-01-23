from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="p2p",
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
        bibtex_citation="""@inproceedings{wrzalik-krechel-2021-gerdalir,
    title = "{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval",
    author = "Wrzalik, Marco  and
      Krechel, Dirk",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.13",
    pages = "123--128",
    abstract = "We present GerDaLIR, a German Dataset for Legal Information Retrieval based on case documents from the open legal information platform Open Legal Data. The dataset consists of 123K queries, each labelled with at least one relevant document in a collection of 131K case documents. We conduct several baseline experiments including BM25 and a state-of-the-art neural re-ranker. With our dataset, we aim to provide a standardized benchmark for German LIR and promote open research in this area. Beyond that, our dataset comprises sufficient training data to be used as a downstream task for German or multilingual language models.",
}""",
    )
