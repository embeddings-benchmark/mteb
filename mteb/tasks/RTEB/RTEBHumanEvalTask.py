from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBHumanEval(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBHumanEval",
        "description": "RTEB evaluation for HumanEval dataset.",
        "reference": "https://github.com/openai/human-eval",
        "dataset_path": "openai/openai_humaneval",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2021-01-01", "2021-12-31"),
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "mit",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{chen2021evaluating,
  title={Evaluating large language models trained on code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Schoelkopf, Qinyuan and Le, Shi Yusong and Stevens, Foster and Ray, Aditya and Puri, Vijay and Agarwal, Rishabh and Fernandez, Lazar and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn", "python-Code"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="HumanEval", **kwargs)
