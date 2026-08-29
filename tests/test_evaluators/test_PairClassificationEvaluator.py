import numpy as np
import pytest
from datasets import Dataset

import mteb
from mteb._evaluators import PairClassificationEvaluator
from mteb.mocks.mock_tasks import MockPairClassificationTask
from mteb.timing import TimingStack

TOL = 0.0001


class TestPairClassificationEvaluator:
    def test_accuracy(self):
        task = MockPairClassificationTask()
        task.load_data()

        evaluator = PairClassificationEvaluator(
            task.dataset["test"],
            input1_column_name="sentence1",
            input2_column_name="sentence2",
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="test",
            input1_prompt_type=None,
            input2_prompt_type=None,
            timer=TimingStack(),
        )
        distances = evaluator(
            mteb.get_model("mteb/baseline-random-encoder"),
            encode_kwargs={"batch_size": 32},
        )
        assert distances["cosine_scores"] == pytest.approx(
            [0.7375020980834961, 0.7731508016586304], TOL
        )
        assert distances["euclidean_distances"] == pytest.approx(
            [2.4108424186706543, 2.1905980110168457], TOL
        )
        assert distances["manhattan_distances"] == pytest.approx(
            [11.177837371826172, 10.721406936645508], TOL
        )
        assert distances["similarity_scores"] == pytest.approx(
            [0.7375020384788513, 0.7731509208679199], TOL
        )
        assert distances["dot_scores"] == pytest.approx(
            [7.974165916442871, 8.176445960998535], TOL
        )

    def test_encodes_unique_items_when_id_columns_are_provided(self):
        task = MockPairClassificationTask()
        dataset = Dataset.from_dict(
            {
                "sentence1": ["a", "a", "b"],
                "sentence1_id": ["video-1", "video-1", "video-2"],
                "sentence2": ["x", "y", "x"],
                "sentence2_id": ["image-1", "image-2", "image-1"],
            }
        )

        class RecordingModel:
            def __init__(self) -> None:
                self.encoded_values: list[list[str]] = []

            def encode(self, inputs, **kwargs):
                values = inputs.dataset["text"]
                self.encoded_values.append(values)
                return np.asarray(
                    [[ord(value), len(value) + 1] for value in values],
                    dtype=np.float32,
                )

        model = RecordingModel()
        evaluator = PairClassificationEvaluator(
            dataset,
            input1_column_name="sentence1",
            input2_column_name="sentence2",
            input1_id_column_name="sentence1_id",
            input2_id_column_name="sentence2_id",
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="test",
            input1_prompt_type=None,
            input2_prompt_type=None,
            timer=TimingStack(),
        )

        distances = evaluator(model, encode_kwargs={"batch_size": 32})

        full_model = RecordingModel()
        full_evaluator = PairClassificationEvaluator(
            dataset,
            input1_column_name="sentence1",
            input2_column_name="sentence2",
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="test",
            input1_prompt_type=None,
            input2_prompt_type=None,
            timer=TimingStack(),
        )
        full_distances = full_evaluator(full_model, encode_kwargs={"batch_size": 32})

        assert model.encoded_values == [["a", "b"], ["x", "y"]]
        assert full_model.encoded_values == [["a", "a", "b"], ["x", "y", "x"]]
        assert all(len(scores) == len(dataset) for scores in distances.values())
        for score_name, scores in distances.items():
            assert scores == pytest.approx(full_distances[score_name])
