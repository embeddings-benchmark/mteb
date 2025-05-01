from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBWikiSQL(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBWikiSQL",
        "description": "RTEB evaluation for WikiSQL dataset.",
        "reference": "https://huggingface.co/datasets/Salesforce/wikisql",
        "dataset_path": "Salesforce/wikisql",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # WikiSQL doesn't specify a date range
        "domains": ["Programming"],
        "task_subtypes": ["Question answering"],
        "license": "not specified",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="WikiSQL", **kwargs)
