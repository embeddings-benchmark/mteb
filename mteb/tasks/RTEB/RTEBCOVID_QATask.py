from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBCOVID_QA(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBCOVID_QA",
        "description": "RTEB evaluation for COVID_QA dataset.",
        "reference": "https://aclanthology.org/2020.nlpcovid19-acl.18/",
        "dataset_path": "castorini/covid_qa_castorini",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2020-01-01", "2020-12-31"),
        "domains": ["Medical"],
        "task_subtypes": ["Question answering"],
        "license": "apache-2.0",
        "annotations_creators": "expert-annotated",
        "text_creation": "found",
        "bibtex_citation": """@inproceedings{moller-etal-2020-covid,
    title = "{COVID}-QA: A Question Answering Dataset for {COVID}-19",
    author = "M{\"o}ller, Erik  and
      Brasch, Malte  and
      Eger, Steffen  and
      {\"U}z{\"u}mc{\"u}o{\\u{g}}lu, Hakan  and
      Reimers, Nils  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 1st Workshop on NLP for COVID-19 (part 2) at ACL 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlpcovid19-acl.18",
    doi = "10.18653/v1/2020.nlpcovid19-acl.18",
    pages = "145--152",
    abstract = "We present COVID-QA, a Question Answering dataset consisting of 2,019 question/answer pairs annotated by volunteer biomedical experts on scientific articles about COVID-19. The dataset is designed to be challenging for current QA systems, as it requires reasoning over multiple sentences and paragraphs. We provide baseline results using several state-of-the-art QA models and analyze their performance.",
}""",
        "modalities": ["text"],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="COVID_QA", **kwargs)
