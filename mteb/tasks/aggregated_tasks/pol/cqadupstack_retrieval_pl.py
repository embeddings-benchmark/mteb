from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
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

task_list_cqa = [
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
@inproceedings{wojtasik2024beirpl,
  address = {Torino, Italia},
  archiveprefix = {arXiv},
  author = {Wojtasik, Konrad and Wo{\l}owiec, Kacper and Shishkin, Vadim and Janz, Arkadiusz and Piasecki, Maciej},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation ({LREC}-{COLING} 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  eprint = {2305.19840},
  month = {may},
  pages = {2149--2160},
  publisher = {ELRA and ICCL},
  title = {{BEIR}-{PL}: Zero Shot Information Retrieval Benchmark for the {P}olish Language},
  url = {https://aclanthology.org/2024.lrec-main.194/},
  year = {2024},
}
""",
        adapted_from=["CQADupstackRetrieval"],
    )
