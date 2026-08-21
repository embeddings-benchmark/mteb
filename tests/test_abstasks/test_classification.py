from datasets import Dataset

from mteb.mocks.mock_tasks import MockClassificationTask


def test_undersample_reads_only_label_column(monkeypatch):
    dataset = Dataset.from_dict(
        {
            "text": [f"sample {i}" for i in range(8)],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    task = MockClassificationTask()
    task.samples_per_label = 2

    original_getitem = Dataset.__getitem__

    def fail_on_row_access(self, key):
        if isinstance(key, int) and "text" in self.column_names:
            raise AssertionError("undersampling should not materialize input rows")
        return original_getitem(self, key)

    monkeypatch.setattr(Dataset, "__getitem__", fail_on_row_access)

    sampled, _, sampled_idxs = task._undersample_data(dataset, experiment_num=0)

    assert len(sampled) == 4
    assert len(sampled_idxs) == 4
    assert sorted(sampled["label"]) == [0, 0, 1, 1]
