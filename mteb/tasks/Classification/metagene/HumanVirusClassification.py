from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HumanVirusClassificationOne(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanVirusClassificationOne",
        description="A Classification task for predicting the virus infecting a human host based on the metagenomics.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanVirusInfecting/tree/main/hv/1",
        dataset={
            "path": "metagene-ai/HumanVirusInfecting",
            "name": "class-1",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
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
        full_train_dataset = self.dataset['train']

        from collections import Counter
        label_counts = Counter(full_train_dataset['source'])

        # by default, we undersample the training data to 8 samples per label
        desired_train_samples = 8

        M = min(label_counts.values())
        if M < desired_train_samples:
            raise ValueError(
                f"Not enough samples per label to achieve {desired_train_samples} "
                f"training samples. The smallest label has only {M} samples."
            )
        test_size = 1 - (desired_train_samples / M)
        split_datasets = full_train_dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42)
        new_train_dataset = split_datasets['train']
        new_test_dataset = split_datasets['test']

        print(f"\nSplitting the data with test_size={test_size}")
        print(f"Train set size: {len(new_train_dataset)} rows")
        print(f"Test set size: {len(new_test_dataset)} rows\n")

        self.dataset = datasets.DatasetDict({
            'train': new_train_dataset,
            'test': new_test_dataset
        })

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "source": "label"})


class HumanVirusClassificationTwo(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanVirusClassificationTwo",
        description="A Classification task for predicting the virus infecting a human host based on the metagenomics.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanVirusInfecting/tree/main/hv/2",
        dataset={
            "path": "metagene-ai/HumanVirusInfecting",
            "name": "class-2",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
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
        full_train_dataset = self.dataset['train']

        from collections import Counter
        label_counts = Counter(full_train_dataset['source'])

        # by default, we undersample the training data to 8 samples per label
        desired_train_samples = 8

        M = min(label_counts.values())
        if M < desired_train_samples:
            raise ValueError(
                f"Not enough samples per label to achieve {desired_train_samples} "
                f"training samples. The smallest label has only {M} samples."
            )
        test_size = 1 - (desired_train_samples / M)
        split_datasets = full_train_dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42)
        new_train_dataset = split_datasets['train']
        new_test_dataset = split_datasets['test']

        print(f"\nSplitting the data with test_size={test_size}")
        print(f"Train set size: {len(new_train_dataset)} rows")
        print(f"Test set size: {len(new_test_dataset)} rows\n")

        self.dataset = datasets.DatasetDict({
            'train': new_train_dataset,
            'test': new_test_dataset
        })

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "source": "label"})


class HumanVirusClassificationThree(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanVirusClassificationThree",
        description="A Classification task for predicting the virus infecting a human host based on the metagenomics.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanVirusInfecting/tree/main/hv/3",
        dataset={
            "path": "metagene-ai/HumanVirusInfecting",
            "name": "class-3",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
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
        full_train_dataset = self.dataset['train']

        from collections import Counter
        label_counts = Counter(full_train_dataset['source'])

        # by default, we undersample the training data to 8 samples per label
        desired_train_samples = 8

        M = min(label_counts.values())
        if M < desired_train_samples:
            raise ValueError(
                f"Not enough samples per label to achieve {desired_train_samples} "
                f"training samples. The smallest label has only {M} samples."
            )
        test_size = 1 - (desired_train_samples / M)
        split_datasets = full_train_dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42)
        new_train_dataset = split_datasets['train']
        new_test_dataset = split_datasets['test']

        print(f"\nSplitting the data with test_size={test_size}")
        print(f"Train set size: {len(new_train_dataset)} rows")
        print(f"Test set size: {len(new_test_dataset)} rows\n")

        self.dataset = datasets.DatasetDict({
            'train': new_train_dataset,
            'test': new_test_dataset
        })

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "source": "label"})


class HumanVirusClassificationFour(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanVirusClassificationFour",
        description="A Classification task for predicting the virus infecting a human host based on the metagenomics.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanVirusInfecting/tree/main/hv/4",
        dataset={
            "path": "metagene-ai/HumanVirusInfecting",
            "name": "class-4",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
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
        full_train_dataset = self.dataset['train']

        from collections import Counter
        label_counts = Counter(full_train_dataset['source'])

        # by default, we undersample the training data to 8 samples per label
        desired_train_samples = 8

        M = min(label_counts.values())
        if M < desired_train_samples:
            raise ValueError(
                f"Not enough samples per label to achieve {desired_train_samples} "
                f"training samples. The smallest label has only {M} samples."
            )
        test_size = 1 - (desired_train_samples / M)
        split_datasets = full_train_dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42)
        new_train_dataset = split_datasets['train']
        new_test_dataset = split_datasets['test']

        print(f"\nSplitting the data with test_size={test_size}")
        print(f"Train set size: {len(new_train_dataset)} rows")
        print(f"Test set size: {len(new_test_dataset)} rows\n")

        self.dataset = datasets.DatasetDict({
            'train': new_train_dataset,
            'test': new_test_dataset
        })

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "source": "label"})