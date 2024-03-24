from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoMrtydi(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Ko-mrtydi",
        description="Ko-mrtydi",
        reference=None,
        hf_hub_name="taeminlee/Ko-mrtydi",
        type="Retrieval",
        category="s2p",
        eval_splits=["dev"],
        eval_langs=["ko"],
        main_score="ndcg_at_10",
        revision="71a2e011a42823051a2b4eb303a3366bdbe048d3",
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
