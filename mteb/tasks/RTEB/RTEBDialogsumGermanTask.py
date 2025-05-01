from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBDialogsumGerman(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBDialogsumGerman",
        "description": "RTEB evaluation for DialogsumGerman dataset.",
        "reference": "https://aclanthology.org/2021.findings-acl.449/",
        "dataset_path": "fathyshalab/Dialogsum-german",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2021-05-01", "2021-05-31"),
        "domains": ["Spoken"],
        "task_subtypes": ["Conversational retrieval"],
        "license": "not specified",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@inproceedings{chen-etal-2021-dialogsum,
    title = "{D}ialog{S}um: A Real-Life Scenario Dialogue Summarization Dataset",
    author = "Chen, Yulong  and
      Liu, Chong  and
      Chen, Xin  and
      Zhao, Hao  and
      Liu, Tianyu  and
      Li, Leyang  and
      Rui, Ruyi  and
      Zhou, Dandan  and
      Wang, Chen  and
      Li, Xiang  and
      Sun, Zheng  and
      Yan, Xiaoyu  and
      Wang, Xixin  and
      Gao, Xin  and
      Yan, Xiang  and
      Huang, Xiaofei  and
      Yan, Huajian  and
      Wang, Xinsong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.449",
    doi = "10.18653/v1/2021.findings-acl.449",
    pages = "5062--5074",
    abstract = "Dialogue summarization is a challenging task that requires understanding the context and generating a concise summary of a conversation. Existing datasets for dialogue summarization are limited in size and diversity, which hinders the development of robust models. In this paper, we propose DialogSum, a large-scale dialogue summarization dataset consisting of 13,460 dialogues with corresponding manually labeled summaries and topics. We collect dialogues from various real-life scenarios, including customer service, online forums, and daily conversations. We also provide a detailed analysis of the dataset and baseline results using state-of-the-art models. Experimental results show that DialogSum is a challenging dataset and provides a valuable resource for future research on dialogue summarization.",
}""",
        "modalities": ["text"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="DialogsumGerman", **kwargs)
