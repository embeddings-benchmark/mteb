from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBChatDoctor_HealthCareMagic(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBChatDoctor_HealthCareMagic",
        "description": "RTEB evaluation for ChatDoctor_HealthCareMagic dataset.",
        "reference": "https://github.com/Kent0n-Li/ChatDoctor",
        "dataset_path": "lavita/ChatDoctor-HealthCareMagic-100k",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2023-06-24", "2023-06-24"),
        "task_subtypes": [],
        "license": "cc-by-4.0",
        "annotations_creators": "derived",
        "text_creation": "found",
        "bibtex_citation": """@article{Li2023ChatDoctor,
  author = {Li, Yunxiang and Li, Zihan and Zhang, Kai and Dan, Ruilong and Jiang, Steve and Zhang, You},
  title = {ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge},
  journal = {Cureus},
  year = {2023},
  volume = {15},
  number = {6},
  pages = {e40895},
  doi = {10.7759/cureus.40895}
}""",
        "modalities": ["text"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="ChatDoctor_HealthCareMagic",
            **kwargs,
        )
