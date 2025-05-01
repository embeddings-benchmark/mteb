from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBMBPP(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBMBPP",
        "description": "RTEB evaluation for MBPP dataset.",
        "reference": "https://huggingface.co/datasets/Muennighoff/mbpp",
        "dataset_path": "Muennighoff/mbpp",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": None,  # MBPP doesn't have a specific date range
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "cc-by-sa-4.0",  # Standardized license format
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{appel2022mbpp,
  title={MBPP: A Code Generation Benchmark for the Classroom},
  author={Appel, Alexander and Yang, Ke and Yin, Pengcheng and others},
  journal={arXiv preprint arXiv:2208.05317},
  year={2022}
}""",
        "modalities": ["text"],
        "eval_langs": ["eng-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="MBPP", **kwargs)
