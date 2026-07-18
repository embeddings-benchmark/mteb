from datasets import Dataset, DatasetDict

from tests.mock_tasks import MockSTSTask


def test_sts_statistics_use_order_insensitive_pairs_across_splits() -> None:
    task = MockSTSTask()
    task.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "sentence1": ["alpha"],
                    "sentence2": ["beta"],
                    "score": [1.0],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "sentence1": ["beta", "gamma", "delta"],
                    "sentence2": ["alpha", "delta", "gamma"],
                    "score": [1.0, 0.5, 0.5],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "sentence1": ["delta", "epsilon"],
                    "sentence2": ["gamma", "zeta"],
                    "score": [0.5, 0.0],
                }
            ),
        }
    )
    task.data_loaded = True

    stats = task._calculate_descriptive_statistics_from_split("validation")

    assert stats["num_samples"] == 3
    assert stats["unique_pairs"] == 2
    assert stats["pair_overlap"] == {"train": 1, "test": 1}
