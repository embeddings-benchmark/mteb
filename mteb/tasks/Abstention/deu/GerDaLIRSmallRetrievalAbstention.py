from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Retrieval.deu.GerDaLIRSmallRetrieval import GerDaLIRSmall


class GerDaLIRSmallAbstention(AbsTaskAbstention, GerDaLIRSmall):
    abstention_task = "Retrieval"
    metadata = TaskMetadata(
        name="GerDaLIRSmallAbstention",
        description="The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "mteb/GerDaLIRSmall",
            "revision": "48327de6ee192e9610f3069789719788957c7abd",
        },
        type="Abstention",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="MIT license",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
