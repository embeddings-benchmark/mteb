from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFrenchOpenFiscalTexts(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBFrenchOpenFiscalTexts",
        "description": "RTEB evaluation for FrenchOpenFiscalTexts dataset.",
        "reference": "https://echanges.dila.gouv.fr/OPENDATA/JADE/",  # OPENDATA/JADE source
        "dataset_path": "StanBienaives/french-open-fiscal-texts",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": (
            "2000-01-01",
            "2023-12-31",
        ),  # Assuming a broad date range for case law data
        "domains": ["Legal", "Financial"],
        "task_subtypes": ["Article retrieval"],
        "license": "not specified",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """unknown""",
        "modalities": ["text"],
        "eval_langs": ["fra-Latn"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="FrenchOpenFiscalTexts",
            **kwargs,
        )
