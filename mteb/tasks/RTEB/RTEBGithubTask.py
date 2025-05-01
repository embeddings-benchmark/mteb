from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBGithub(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBGithub",
        "description": "RTEB evaluation for Github dataset.",
        "reference": "https://github.com/CoIR-team/coir",
        "dataset_path": "CoIR-team/Github",  # Updated from TODO placeholder
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2024-07-03", "2024-07-03"),
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "apache-2.0",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """@misc{li2024coircomprehensivebenchmarkcode,
      title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Hao Zhang and Xinyi Dai and Yasheng Wang and Ruiming Tang},
      year={2024},
      eprint={2407.02883},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.02883}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn", "python-Code"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="Github", **kwargs)
