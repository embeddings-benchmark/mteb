from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Retrieval.eng.SciFactRetrieval import SciFact


class SciFactAbstention(AbsTaskAbstention, SciFact):
    abstention_task = "Retrieval"
    metadata = TaskMetadata(
        name="SciFactAbstention",
        dataset={
            "path": "mteb/scifact",
            "revision": "0228b52cf27578f30900b9e5271d331663a030d7",
        },
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Abstention",
        category="s2p",
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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
