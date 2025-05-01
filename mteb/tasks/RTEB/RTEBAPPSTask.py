from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBAPPS(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBAPPS",
        "description": "RTEB evaluation for APPS dataset.",
        "reference": "https://arxiv.org/abs/2105.09938",
        "dataset_path": "CoIR-Retrieval/apps",
        "dataset_revision": "f22508f96b7a36c2415181ed8bb76f76e04ae2d5",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2021-05-20", "2021-05-20"),
        "task_subtypes": ["Code retrieval"],
        "license": "mit",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}""",
        "modalities": ["text"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="APPS", **kwargs)
