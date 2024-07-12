from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackTexRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackTexRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "mteb/cqadupstack-tex",
            "revision": "46989137a86843e03a6195de44b09deda022eec7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{hoogeveen2015,
author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
series = {ADCS '15},
year = {2015},
isbn = {978-1-4503-4040-3},
location = {Parramatta, NSW, Australia},
pages = {3:1--3:8},
articleno = {3},
numpages = {8},
url = {http://doi.acm.org/10.1145/2838931.2838934},
doi = {10.1145/2838931.2838934},
acmid = {2838934},
publisher = {ACM},
address = {New York, NY, USA},
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1297.09043177285,
                    "average_query_length": 46.935306262904334,
                    "num_documents": 68184,
                    "num_queries": 2906,
                    "average_relevant_docs_per_query": 1.7735719201651754,
                }
            },
        },
    )
