from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBAILAStatutes(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBAILAStatutes",
        "description": "RTEB evaluation for AILAStatutes dataset.",
        "reference": "https://zenodo.org/records/4063986",
        "dataset_path": "zenodo/4063986",  # Using Zenodo DOI as path
        "dataset_revision": "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
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
        super().__init__(rteb_dataset_name="AILAStatutes", **kwargs)
