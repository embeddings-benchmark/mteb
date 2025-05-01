from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBHC3Finance(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBHC3Finance",
        "description": "RTEB evaluation for HC3Finance dataset.",
        "reference": "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        "dataset_path": "Atharva07/hc3_finance",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # Original dataset had date range (2023-01-01 to 2023-12-31) but set to None for consistency
        "domains": ["Financial"],
        "task_subtypes": ["Question answering"],
        "license": "not specified",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{guo2023towards,
  title={Towards a Human-ChatGPT Comparative Corpus on Question Answering},
  author={Guo, Jiaxin and Fan, Kai and Su, Xin and Gao, Jundong and Ji, Shuo and Zhou, Yuquan and Wu, Xuejie and Wang, Cong},
  journal={arXiv preprint arXiv:2301.13867},
  year={2023}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="HC3Finance", **kwargs)
