from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFrenchTriviaQAWikicontext(AbsTaskRTEB):
    _TASK_SPECIFIC_METADATA = {
        "task_name": "RTEBFrenchTriviaQAWikicontext",
        "description": "RTEB evaluation for FrenchTriviaQAWikicontext dataset.",
        "reference": "https://www.cs.utexas.edu/~eunsol/files/papers/acl17jcwz.pdf",
        "dataset_path": "manu/french-trivia",
        "dataset_revision": "main",
        "main_score": "ndcg_at_10",
        "revision": "1.0.1",
        "date": ("2017-01-01", "2017-12-31"),
        "domains": ["Spoken"],
        "task_subtypes": ["Question answering"],
        "license": "not specified",
        "annotations_creators": "human-annotated",
        "text_creation": "found",
        "bibtex_citation": """@article{joshi2017triviaqa,
  title={TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension},
  author={Joshi, Mandar and Choi, Eunsol and Weld, Daniel S and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1705.03565},
  year={2017}
}""",
        "modalities": ["text"],
        "eval_langs": ["fra-Latn"],
        "dialect": [],
    }

    metadata = AbsTaskRTEB.create_rteb_task_metadata(**_TASK_SPECIFIC_METADATA)

    def __init__(self, **kwargs):
        super().__init__(
            rteb_dataset_name="FrenchTriviaQAWikicontext",
            **kwargs,
        )
