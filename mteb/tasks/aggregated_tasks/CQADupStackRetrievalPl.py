from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.Retrieval.pol.CqadupstackPLRetrieval import (
    CQADupstackAndroidRetrievalPL,
    CQADupstackEnglishRetrievalPL,
    CQADupstackGamingRetrievalPL,
    CQADupstackGisRetrievalPL,
    CQADupstackMathematicaRetrievalPL,
    CQADupstackPhysicsRetrievalPL,
    CQADupstackProgrammersRetrievalPL,
    CQADupstackStatsRetrievalPL,
    CQADupstackTexRetrievalPL,
    CQADupstackUnixRetrievalPL,
    CQADupstackWebmastersRetrievalPL,
    CQADupstackWordpressRetrievalPL,
)

task_list_cqa: list[AbsTask] = [
    CQADupstackAndroidRetrievalPL(),
    CQADupstackEnglishRetrievalPL(),
    CQADupstackGamingRetrievalPL(),
    CQADupstackGisRetrievalPL(),
    CQADupstackMathematicaRetrievalPL(),
    CQADupstackPhysicsRetrievalPL(),
    CQADupstackProgrammersRetrievalPL(),
    CQADupstackStatsRetrievalPL(),
    CQADupstackTexRetrievalPL(),
    CQADupstackUnixRetrievalPL(),
    CQADupstackWebmastersRetrievalPL(),
    CQADupstackWordpressRetrievalPL(),
]


class CQADupstackRetrievalPL(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="CQADupstackRetrieval-PL",
        description="CQADupstackRetrieval-PL",
        reference="",
        tasks=task_list_cqa,
        main_score="ndcg_at_10",
        type="Retrieval",  # since everything is retrieval - otherwise it would be "Aggregated"
        eval_splits=["test"],
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )
