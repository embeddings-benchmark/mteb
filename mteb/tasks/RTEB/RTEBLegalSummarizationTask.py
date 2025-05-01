from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBLegalSummarization(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBLegalSummarization",
        "description": "RTEB evaluation for LegalSummarization dataset.",
        "reference": "https://huggingface.co/datasets/mteb/legal_summarization",
        "dataset_path": "mteb/legal_summarization",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # No specific date range available
        "domains": ["Legal"],
        "task_subtypes": ["Article retrieval"],
        "license": "cc-by-sa-4.0",  # Standardized license format
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="LegalSummarization",
            **kwargs,
        )
