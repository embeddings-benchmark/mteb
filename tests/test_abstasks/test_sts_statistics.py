from datasets import Dataset, DatasetDict

from tests.mock_tasks import MockMultilingualSTSTask, MockSTSTask


def test_sts_statistics_count_order_insensitive_pairs() -> None:
    task = MockSTSTask()
    task.dataset = DatasetDict(
        {
            "validation": Dataset.from_dict(
                {
                    "sentence1": ["beta", "gamma", "delta"],
                    "sentence2": ["alpha", "delta", "gamma"],
                    "score": [1.0, 0.5, 0.5],
                }
            ),
        }
    )
    task.data_loaded = True

    stats = task._calculate_descriptive_statistics_from_split("validation")

    assert stats["num_samples"] == 3
    assert stats["unique_pairs"] == 2


def test_sts_overall_statistics_count_pairs_across_subsets() -> None:
    task = MockMultilingualSTSTask()
    task.dataset = {
        "eng": DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": ["alpha"],
                        "sentence2": ["beta"],
                        "score": [1.0],
                    }
                )
            }
        ),
        "fra": DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": ["beta"],
                        "sentence2": ["alpha"],
                        "score": [1.0],
                    }
                )
            }
        ),
    }
    task.data_loaded = True

    stats = task._calculate_descriptive_statistics_from_split(
        "test", compute_overall=True
    )

    assert stats["num_samples"] == 2
    assert stats["unique_pairs"] == 1
