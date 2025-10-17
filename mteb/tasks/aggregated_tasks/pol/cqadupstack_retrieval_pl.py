from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.retrieval.pol.cqadupstack_pl_retrieval import (
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
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["CQADupstackRetrieval"],
    )
