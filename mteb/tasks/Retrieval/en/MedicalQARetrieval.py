from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedicalQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedicalQA",
        description="The dataset consistes 16.4 medical question and answer pairs.",
        reference="https://github.com/lauramanor/legal_summarization",
        dataset={
            "path": "Sakshamrzt/medical_qa",
            "revision": "7b14a4ab92cabc4d2506d1dd138507f904f0928e",
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
