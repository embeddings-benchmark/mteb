from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoStrategyQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Ko-StrategyQA",
        description="Ko-StrategyQA",
        reference=None,
        hf_hub_name="taeminlee/Ko-StrategyQA",
        type="Retrieval",
        category="s2p",
        eval_splits=["dev"],
        eval_langs=["ko"],
        main_score="ndcg_at_10",
        revision="d243889a3eb6654029dbd7e7f9319ae31d58f97c",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
