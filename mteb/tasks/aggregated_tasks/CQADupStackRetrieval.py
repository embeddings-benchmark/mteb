from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.Retrieval import (
    CQADupstackAndroidRetrieval,
    CQADupstackEnglishRetrieval,
    CQADupstackGamingRetrieval,
    CQADupstackGisRetrieval,
    CQADupstackMathematicaRetrieval,
    CQADupstackPhysicsRetrieval,
    CQADupstackProgrammersRetrieval,
    CQADupstackStatsRetrieval,
    CQADupstackTexRetrieval,
    CQADupstackUnixRetrieval,
    CQADupstackWebmastersRetrieval,
    CQADupstackWordpressRetrieval,
)

task_list_cqa: list[AbsTask] = [
    CQADupstackAndroidRetrieval(),
    CQADupstackEnglishRetrieval(),
    CQADupstackGamingRetrieval(),
    CQADupstackGisRetrieval(),
    CQADupstackMathematicaRetrieval(),
    CQADupstackPhysicsRetrieval(),
    CQADupstackProgrammersRetrieval(),
    CQADupstackStatsRetrieval(),
    CQADupstackTexRetrieval(),
    CQADupstackUnixRetrieval(),
    CQADupstackWebmastersRetrieval(),
    CQADupstackWordpressRetrieval(),
]


class CQADupstackRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="CQADupstackRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        tasks=task_list_cqa,
        main_score="ndcg_at_10",
        type="Retrieval",  # since everything is retrieval - otherwise it would be "Aggregated"
        eval_splits=["test"],
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
    )
