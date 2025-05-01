from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFiQAPersonalFinance(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBFiQAPersonalFinance",
        "description": "RTEB evaluation for FiQAPersonalFinance dataset.",
        "reference": "https://sites.google.com/view/fiqa/home",
        "dataset_path": "bilalRahib/fiqa-personal-finance-dataset",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2018-01-01", "2018-12-31"),
        "domains": ["Financial"],
        "task_subtypes": ["Question answering"],
        "license": "not specified",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@inproceedings{fiqa_2018,
    title = {{FiQA-2018} Shared Task: Financial Opinion Mining and Question Answering},
    author = {Radu Tudor Ionescu and Saif Mohammad and Svetlana Kiritchenko and Smaranda Muresan},
    booktitle = {Proceedings of the {ACL} 2018 Workshop on Building {NLP} Solutions for Under Resourced Languages ({BNSUL})},
    month = jul,
    year = {2018},
    address = {Melbourne, Australia},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/W18-3501},
    doi = {10.18653/v1/W18-3501},
    pages = {1--10}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="FiQAPersonalFinance",
            **kwargs,
        )
