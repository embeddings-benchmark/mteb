from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBJapanLaw(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBJapanLaw",
        "description": "RTEB evaluation for JapanLaw dataset.",
        "reference": "https://huggingface.co/datasets/y2lan/japan-law",
        "dataset_path": "TODO/JapanLaw",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,
        "domains": ["Legal"],
        "task_subtypes": ["Article retrieval"],
        "license": "mit",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": ["jpn-Jpan"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="JapanLaw", **kwargs)
