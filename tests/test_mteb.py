from __future__ import annotations

import logging
from typing import Union

import pytest
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.tasks.BitextMining.dan.BornholmskBitextMining import BornholmBitextMining

logging.basicConfig(level=logging.INFO)


def test_two_mteb_tasks():
    """Test that two tasks can be fetched and run"""
    model = SentenceTransformer("average_word_embeddings_komninos")
    eval = MTEB(
        tasks=[
            "STS12",
            "SummEval",
        ]
    )
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize(
    "task",
    [
        BornholmBitextMining(),
        "TwentyNewsgroupsClustering",
        "TwentyNewsgroupsClustering.v2",
        "Banking77Classification",
        "SciDocsRR",
        "SprintDuplicateQuestions",
        "NFCorpus",
        "STS12",
        "SummEval",
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "average_word_embeddings_levy_dependency",
    ],
)
def test_mteb_task(task: Union[str, AbsTask], model_name: str):
    """Test that a task can be fetched and run"""
    model = SentenceTransformer(model_name)
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)


def test_all_tasks_fetch():
    """Test that all tasks can be fetched"""
    MTEB.mteb_tasks()
