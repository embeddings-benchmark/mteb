from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBJapaneseCoNaLa(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBJapaneseCoNaLa",
        "description": "RTEB evaluation for JapaneseCoNaLa dataset.",
        "reference": "https://huggingface.co/datasets/haih2/japanese-conala",
        "dataset_path": "haih2/japanese-conala",
        "dataset_revision": "main",  # Assuming main based on HF page
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "not specified",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": [
            "jpn-Jpan",
            "python-Code",
        ],  # Including python-Code as it's a code generation dataset
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="JapaneseCoNaLa", **kwargs)
