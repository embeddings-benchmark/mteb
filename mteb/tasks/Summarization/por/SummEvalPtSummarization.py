from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalPtSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalPtSummarization",
        description="Portugese Summarization Dataset",
        reference="",
        dataset={
            "path": "mteb-pt/summeval",
            "revision": "b57d536d89da6fc42b3ca40645bb813f538b9ede",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=["written"],
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

