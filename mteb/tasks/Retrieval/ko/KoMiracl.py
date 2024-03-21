from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoMiracl(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Ko-miracl",
        description="Ko-miracl",
        reference=None,
        hf_hub_name="taeminlee/Ko-miracl",
        type="Retrieval",
        category="s2p",
        eval_splits=["dev"],
        eval_langs=["ko"],
        main_score="ndcg_at_10",
        revision="5c7690518e481375551916f24241048cf7b017d0",
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
        n_samples={},
        avg_character_length={},
    )
