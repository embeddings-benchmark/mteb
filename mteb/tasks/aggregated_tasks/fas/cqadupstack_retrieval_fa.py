from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.retrieval import (
    CQADupstackAndroidRetrievalFa,
    CQADupstackEnglishRetrievalFa,
    CQADupstackGamingRetrievalFa,
    CQADupstackGisRetrievalFa,
    CQADupstackMathematicaRetrievalFa,
    CQADupstackPhysicsRetrievalFa,
    CQADupstackProgrammersRetrievalFa,
    CQADupstackStatsRetrievalFa,
    CQADupstackTexRetrievalFa,
    CQADupstackUnixRetrievalFa,
    CQADupstackWebmastersRetrievalFa,
    CQADupstackWordpressRetrievalFa,
)

task_list_cqa = [
    CQADupstackAndroidRetrievalFa(),
    CQADupstackEnglishRetrievalFa(),
    CQADupstackGamingRetrievalFa(),
    CQADupstackGisRetrievalFa(),
    CQADupstackMathematicaRetrievalFa(),
    CQADupstackPhysicsRetrievalFa(),
    CQADupstackProgrammersRetrievalFa(),
    CQADupstackStatsRetrievalFa(),
    CQADupstackTexRetrievalFa(),
    CQADupstackUnixRetrievalFa(),
    CQADupstackWebmastersRetrievalFa(),
    CQADupstackWordpressRetrievalFa(),
]


class CQADupstackRetrievalFa(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="CQADupstackRetrieval-Fa",
        description="CQADupstackRetrieval-Fa",
        reference="",
        tasks=task_list_cqa,
        main_score="ndcg_at_10",
        type="Retrieval",  # since everything is retrieval - otherwise it would be "Aggregated"
        eval_splits=["test"],
        bibtex_citation=""" """,
        adapted_from=["CQADupstackRetrieval"],
    )
