# Concrete RTEB task definition for GermanLegalSentences
from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBGermanLegalSentences(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBGermanLegalSentences",
        "description": "RTEB evaluation for GermanLegalSentences dataset.",
        "reference": "http://openlegaldata.io/",  # Open Legal Data source
        "dataset_path": "lavis-nlp/german_legal_sentences",
        "dataset_revision": "main",
        "eval_langs": ["deu-Latn"],
        "main_score": "ndcg_at_10",
        "domains": ["Legal"],
        "task_subtypes": ["Article retrieval"],
        "license": "not specified",  # TODO: Verify license
        "annotations_creators": "LM-generated",
        "text_creation": "found",
        "bibtex_citation": """unknown""",  # TODO: Add bibtex citation
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="GermanLegalSentences",
            **kwargs,
        )
