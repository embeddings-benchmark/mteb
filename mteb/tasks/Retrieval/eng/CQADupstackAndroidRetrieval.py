from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackAndroidRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackAndroidRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "mteb/cqadupstack-android",
            "revision": "f46a197baaae43b4f621051089b82a364682dfeb",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
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
        n_samples=None,
        avg_character_length={
            "test": {
                "average_document_length": 593.701974084703,
                "average_query_length": 51.76680972818312,
                "num_documents": 22998,
                "num_queries": 699,
                "average_relevant_docs_per_query": 2.4263233190271816,
            }
        },
    )
