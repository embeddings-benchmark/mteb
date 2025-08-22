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
