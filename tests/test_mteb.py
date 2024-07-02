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
from mteb.tasks.Classification.multilingual.IndicSentimentClassification import (
    IndicSentimentClassification,
)
from mteb.tasks.Clustering.eng.TwentyNewsgroupsClustering import (
    TwentyNewsgroupsClusteringFast,
)

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


twenty_news = TwentyNewsgroupsClusteringFast()

# downsample to speed up tests
twenty_news.max_document_to_embed = 1000
twenty_news.n_clusters = 2
twenty_news.max_fraction_of_documents_to_embed = None

task_test_cases = [
    BornholmBitextMining(),  # bitext mining + just supplying a task class instead of a string
    IndicSentimentClassification(  # multi subset loader
        hf_subsets=["as"],  # we only load one subset here to speed up tests
        n_experiments=2,  # to speed up the test
    ),
    "TwentyNewsgroupsClustering",  # clustering and string instead of class
    twenty_news,  # fast clustering
    "Banking77Classification",  # classification
    "SciDocsRR",  # reranking
    "FarsTail",  # pair classification
    "TwitterHjerneRetrieval",  # retrieval
    "BrazilianToxicTweetsClassification",  # multilabel classification
    "FaroeseSTS",  # STS
    "SummEval",  # summarization
]

task_test_cases_only_string = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in task_test_cases
]


@pytest.mark.parametrize("task", task_test_cases)
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
    task_test_cases,
)
def test_prompt_name_passed_to_all_encodes(task_name: str | AbsTask):
    """Test that all tasks correctly pass down the task_name to the encoder which supports it, and that the encoder which does not support it does not
    receive it.
    """
    _task_name = (
        task_name.metadata.name if isinstance(task_name, AbsTask) else task_name
    )

    class EncoderWithInstructions(Encoder):
        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert prompt_name == _task_name
            return np.zeros((len(sentences), 10))

    class EncoderWithoutInstructions(SentenceTransformer):
        def encode(self, sentences, **kwargs):
            assert "prompt_name" not in kwargs
            return super().encode(sentences, **kwargs)

    if isinstance(task_name, AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = EncoderWithInstructions()
    eval.run(model, output_folder="tests/results", overwrite_results=True)
    # Test that the task_name is not passed down to the encoder
    model = EncoderWithoutInstructions("average_word_embeddings_levy_dependency")
    assert model.prompts == {}, "The encoder should not have any prompts"
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize(
    "task_name",
    task_test_cases,
)
def test_encode_kwargs_passed_to_all_encodes(task_name: str | AbsTask):
    """Test that all tasks correctly pass down the encode_kwargs to the encoder."""
    my_encode_kwargs = {"no_one_uses_this_args": "but_its_here"}

    class TestEncoder(Encoder):
        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert kwargs == my_encode_kwargs
            return np.zeros((len(sentences), 10))

    if isinstance(task_name, AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = TestEncoder()
    eval.run(
        model,
        output_folder="tests/results",
        overwrite_results=True,
        encode_kwargs=my_encode_kwargs,
    )


def test_all_tasks_fetch():
    """Test that all tasks can be fetched"""
    MTEB.mteb_tasks()
