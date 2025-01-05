from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


# https://github.com/embeddings-benchmark/mteb/blob/ad05983fc3e44afc9087328f010a06ceb83f6f7d/mteb/tasks/MultiLabelClassification/por/BrazilianToxicTweetsClassification.py
class HumanMicrobiomeProjectDemonstrationMultiClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectDemonstrationMultiLabelClassification",
        description="",
        dataset={
            "path": "metagene-ai/HumanMicrobiomeProjectDemonstration",
            "name": "multi-label",
            "revision": "main",
        },
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = "logReg"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        from transformers.trainer_utils import set_seed
        set_seed(42)

        import datasets
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore

        self.dataset_transform()

        full_train_dataset = self.dataset['train']
        test_size = 0.9
        split_datasets = full_train_dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42)
        new_train_dataset = split_datasets['train']
        new_test_dataset = split_datasets['test']
        self.dataset = datasets.DatasetDict({
            'train': new_train_dataset,
            'test': new_test_dataset
        })
        print(f"\nSplitting the data with test_size={test_size}")
        print(f"Train set size: {len(new_train_dataset)} rows")
        print(f"Test set size: {len(new_test_dataset)} rows\n")

        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text"})

        dataset = self.dataset["train"]
        categorical_columns = ["disease", "host_sex", "isolation_source"]

        mappings = {}
        for col in categorical_columns:
            unique_values = dataset.unique(col)
            unique_values = sorted(unique_values)
            value_to_idx = {val: i for i, val in enumerate(unique_values)}
            mappings[col] = value_to_idx

        def encode_example(example):
            for col, mapper in mappings.items():
                example[col] = mapper[example[col]]
            return example
        encoded_dataset = dataset.map(encode_example)
        self.dataset["train"] = encoded_dataset

        n_size = len(self.dataset["train"])
        labels = [[] for _ in range(n_size)]
        for c in categorical_columns:
            col_list = self.dataset["train"][c]
            for i in range(n_size):
                if col_list[i] > 0:
                    labels[i].append(c)
        self.dataset["train"] = self.dataset["train"].add_column("label", labels)
        self.dataset["train"] = self.dataset["train"].remove_columns(categorical_columns)