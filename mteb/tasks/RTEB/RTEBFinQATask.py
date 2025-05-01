from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFinQA(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBFinQA",
        "description": "RTEB evaluation for FinQA dataset.",
        "reference": "https://finqasite.github.io/",
        "dataset_path": "ibm-research/finqa",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # Original dataset had date (2021-09-01) but set to None for consistency
        "domains": ["Financial"],
        "task_subtypes": ["Question answering"],
        "license": "mit",  # Standardized license format
        "annotations_creators": "expert-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Wenhu and Chen, Zhiyu and Wang, Chuhan and Zhang, Xinyi and Zhang, Yuchi and Smrz, Pavel and Yu, Xiangyu and Fung, Pascale},
  journal={arXiv preprint arXiv:2109.00122},
  year={2021}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="FinQA", **kwargs)
