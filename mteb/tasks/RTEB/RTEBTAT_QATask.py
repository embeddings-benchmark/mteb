from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBTAT_QA(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBTAT_QA",
        "description": "RTEB evaluation for TAT_QA dataset.",
        "reference": "https://huggingface.co/datasets/next-tat/TAT-QA",
        "dataset_path": "next-tat/TAT-QA",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # TAT-QA doesn't specify a date range
        "domains": ["Financial"],
        "task_subtypes": ["Question answering"],
        "license": "cc-by-sa-4.0",  # Standardized license format
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="TAT_QA", **kwargs)
