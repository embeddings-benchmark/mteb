from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder
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
        "MalteseNewsClassification",
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


@pytest.mark.parametrize(
    "task_name",
    [
        "BornholmBitextMining",
        "TwentyNewsgroupsClustering",
        "TwentyNewsgroupsClustering.v2",
        "Banking77Classification",
        "SciDocsRR",
        "SprintDuplicateQuestions",
        "NFCorpus",
        "MalteseNewsClassification",
        "STS12",
        "SummEval",
    ],
)
def test_mteb_with_instructions(task_name: str):
    """Test that all tasks correctly pass down the task_name to the encoder which supports it, and that the encoder which does not support it does not
    receive it.
    """

    class EncoderWithInstructions(Encoder):
        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert prompt_name == task_name
            return np.zeros((len(sentences), 10))

    class EncoderWithoutInstructions(SentenceTransformer):
        def encode(self, sentences, **kwargs):
            assert "prompt_name" not in kwargs
            return super().encode(sentences, **kwargs)

    tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = EncoderWithInstructions()
    eval.run(model, output_folder="tests/results", overwrite_results=True)
    # Test that the task_name is not passed down to the encoder
    model = EncoderWithoutInstructions("average_word_embeddings_levy_dependency")
    assert model.prompts == {}, "The encoder should not have any prompts"
    eval.run(model, output_folder="tests/results", overwrite_results=True)


def test_all_tasks_fetch():
    """Test that all tasks can be fetched"""
    MTEB.mteb_tasks()
