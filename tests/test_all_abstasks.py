from __future__ import annotations

import logging
from typing import Union

import pytest
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.tasks.BitextMining.da.BornholmskBitextMining import BornholmBitextMining
from unittest.mock import patch, Mock
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logging.basicConfig(level=logging.INFO)

@pytest.mark.parametrize("task", MTEB().tasks_cls)
@patch("datasets.load_dataset")
def test_load_data(mock_load_dataset: Mock, task: AbsTask):
    # TODO: We skip because this load_data is completely different.
    if isinstance(task, AbsTaskRetrieval):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.is_crosslingual and not task.is_multilingual:
            mock_dataset_transform.assert_called_once()

def test_two_mteb_tasks():
    """
    Test that two tasks can be fetched and run
    """
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
    """
    Test that a task can be fetched and run
    """
    model = SentenceTransformer(model_name)
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)


def test_all_tasks_fetch():
    """
    Test that all tasks can be fetched
    """
    MTEB.mteb_tasks()
