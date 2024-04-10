from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedicalQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedicalQA",
        description="The dataset consists 2048 medical question and answer pairs.",
        reference="https://github.com/lauramanor/legal_summarization",
        dataset={
            "path": "mteb/medical_qa",
            "revision": "ae763399273d8b20506b80cf6f6f9a31a6a2b238",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="CC0 1.0 Universal",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples={"default": 2048},
        avg_character_length=None,
    )
