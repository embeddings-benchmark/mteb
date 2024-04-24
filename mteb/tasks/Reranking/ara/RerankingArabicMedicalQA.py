from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class RerankingArabicMedicalQA(AbsTaskReranking):
    metadata = TaskMetadata(
        name="RerankingArabicMedicalQA",
        description="Arabic MedicalQA Dataset - Arabic medical questions with annotations marking pairs of questions as similar or non-similar",
        reference="https://huggingface.co/datasets/gagan3012/reranking-arabic-medicalqa",
        dataset={
            "path": "gagan3012/reranking-arabic-medicalqa",
            "revision": "6122bba171d838e8394beaea34317625d8ff408e",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ara-Arab"],
        main_score="map",
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
        n_samples={"test": 2048},  # 17552
        avg_character_length={"test": 153.11},
    )
