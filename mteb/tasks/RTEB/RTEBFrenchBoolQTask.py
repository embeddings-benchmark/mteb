from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFrenchBoolQ(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBFrenchBoolQ",
        "description": "RTEB evaluation for FrenchBoolQ dataset.",
        "reference": "https://github.com/google-research-datasets/boolean-questions",
        "dataset_path": "manu/french_boolq",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2019-01-01", "2019-12-31"),
        "domains": ["Spoken"],
        "task_subtypes": ["Question answering"],
        "license": "not specified",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{clark2019boolq,
  title={BoolQ: Exploring the surprising difficulty of natural Yes/No questions},
  author={Clark, Christopher and Lee, Kenton and Chang, Ming-Wei and Kwiatkowski, Tom and Collins, Michael and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1905.10441},
  year={2019}
}""",
        "modalities": ["text"],
        "eval_langs": ["fra-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="FrenchBoolQ", **kwargs)
