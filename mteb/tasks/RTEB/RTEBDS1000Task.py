from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBDS1000(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBDS1000",
        "description": "RTEB evaluation for DS1000 dataset.",
        "reference": "https://ds1000-code-gen.github.io/",
        "dataset_path": "xlangai/DS-1000",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2022-11-18", "2022-11-18"),
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "cc-by-sa-4.0",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{luo2022ds,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Luo, Zhoujun and Wang, Chong and Wang, Shangqing and Xia, Han and Zhang, Yuyao and Yu, Shujie and Yin, Hailian and Li, Shi Han and Lai, Binyuan and Chen, Xuanlin and others},
  journal={arXiv preprint arXiv:2211.11501},
  year={2022}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn", "python-Code"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="DS1000", **kwargs)
