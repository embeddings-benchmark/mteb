from __future__ import annotations

import logging

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Retrieval.fra.FQuADRetrieval import FQuADRetrieval

logger = logging.getLogger(__name__)


class FQuADRetrievalAbstention(AbsTaskAbstention, FQuADRetrieval):
    abstention_task = "Retrieval"
    _EVAL_SPLITS = ["test", "validation"]

    metadata = TaskMetadata(
        name="FQuADRetrievalAbstention",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Abstention",
        category="s2p",
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2019-11-01", "2020-05-01"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{dhoffschmidt-etal-2020-fquad,
    title = "{FQ}u{AD}: {F}rench Question Answering Dataset",
    author = "d{'}Hoffschmidt, Martin  and
      Belblidia, Wacim  and
      Heinrich, Quentin  and
      Brendl{\'e}, Tom  and
      Vidal, Maxime",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.107",
    doi = "10.18653/v1/2020.findings-emnlp.107",
    pages = "1193--1208",
}""",
        n_samples={"test": 400, "validation": 100},
        avg_character_length={"test": 937, "validation": 930},
    )
