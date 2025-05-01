from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBAILACasedocs(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBAILACasedocs",
        "description": "RTEB evaluation for AILACasedocs dataset.",
        "reference": "https://zenodo.org/records/4063986",
        "dataset_path": "zenodo/4063986",  # Using Zenodo DOI as path
        "dataset_revision": "4106e6bcc72e0698d714ea8b101355e3e238431a",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # Date not specified in dataset metadata
        "domains": ["Legal"],
        "task_subtypes": ["Article retrieval"],
        "license": "cc-by-4.0",  # Standardized license format
        "bibtex_citation": """@dataset{paheli_bhattacharya_2020_4063986,
  author       = {Paheli Bhattacharya and
                  Kripabandhu Ghosh and
                  Saptarshi Ghosh and
                  Arindam Pal and
                  Parth Mehta and
                  Arnab Bhattacharya and
                  Prasenjit Majumder},
  title        = {AILA 2019 Precedent & Statute Retrieval Task},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4063986},
  url          = {https://doi.org/10.5281/zenodo.4063986}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        # Allow configuration via environment variable or default to the original path
        super().__init__(rteb_dataset_name="AILACasedocs", **kwargs)
