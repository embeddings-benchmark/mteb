from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBConvFinQA(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBConvFinQA",
        "description": "RTEB evaluation for ConvFinQA dataset.",
        "reference": "https://github.com/czyssrs/ConvFinQA",
        "dataset_path": "FinGPT/fingpt-convfinqa",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2022-10-07", "2022-10-07"),
        "task_subtypes": ["Question answering"],
        "license": "mit",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """@article{chen2022convfinqa,
  title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
  author={Chen, Zhiyu and Chen, Wenhu and Wang, Chuhan and Zhang, Xinyi and Zhang, Yuchi and Smrz, Pavel and Yu, Xiangyu and Fung, Pascale},
  journal={arXiv preprint arXiv:2210.03849},
  year={2022}
}""",
        "modalities": ["text"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="ConvFinQA", **kwargs)
