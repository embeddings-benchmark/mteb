from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBLegalQuAD(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBLegalQuAD",
        "description": "RTEB evaluation for LegalQuAD dataset.",
        "reference": "https://github.com/elenanereiss/LegalQuAD",
        "dataset_path": "elenanereiss/LegalQuAD",  # Updated from local path to HF path
        "dataset_revision": "dd73c838031a4914a7a1a16d785b8cec617aaaa4",
        "main_score": "ndcg_at_10",
        "revision": "1.0.0",
        "date": None,  # LegalQuAD doesn't have a specific date range
        "domains": ["Legal"],
        "task_subtypes": ["Question answering"],
        "license": "cc-by-nc-sa-4.0",  # Standardized license format
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """@inproceedings{reiss-etal-2021-legalquad,
  title={LegalQuAD: A Dataset for Legal Question Answering over Documents},
  author={Reiss, Elena and Wohlfarth, Maximilian and Wirth, Christian and Biemann, Chris},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2021},
  organization={ACL}
}""",
        "modalities": ["text"],
        "eval_langs": ["deu-Latn"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="LegalQuAD", **kwargs)
