from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackGisRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGisRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "mteb/cqadupstack-gis",
            "revision": "5003b3064772da1887988e05400cf3806fe491f2",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Written", "Non-fiction"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hoogeveen2015,
  acmid = {2838934},
  address = {New York, NY, USA},
  articleno = {3},
  author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
  booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
  doi = {10.1145/2838931.2838934},
  isbn = {978-1-4503-4040-3},
  location = {Parramatta, NSW, Australia},
  numpages = {8},
  pages = {3:1--3:8},
  publisher = {ACM},
  series = {ADCS '15},
  title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
  url = {http://doi.acm.org/10.1145/2838931.2838934},
  year = {2015},
}
""",
    )
