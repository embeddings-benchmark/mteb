from __future__ import annotations

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from time import time
from typing import Dict, Tuple

from ....evaluation.evaluators import RetrievalEvaluator

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata


logger = logging.getLogger(__name__)
from ....abstasks.AbsTaskAbstention import AbsTaskAbstention


class FQuADRetrievalAbstention(AbsTaskAbstention):

    _EVAL_SPLITS = ["test", "validation"]

    metadata = TaskMetadata(
        name="FQuADRetrievalAbstention",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        dataset_raw = datasets.load_dataset(
            **self.metadata_dict["dataset"],
        )

        # set valid_hasAns and test_hasAns as the validation and test splits (only queries with answers)
        dataset_raw["validation"] = dataset_raw["valid_hasAns"]
        del dataset_raw["valid_hasAns"]

        dataset_raw["test"] = dataset_raw["test_hasAns"]
        del dataset_raw["test_hasAns"]

        # rename  context column to text
        dataset_raw = dataset_raw.rename_column("context", "text")

        self.queries = {
            eval_split: {
                str(i): q["question"] for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.corpus = {
            eval_split: {str(row["title"]): row for row in dataset_raw[eval_split]}
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.relevant_docs = {
            eval_split: {
                str(i): {str(q["title"]): 1}
                for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.data_loaded = True
