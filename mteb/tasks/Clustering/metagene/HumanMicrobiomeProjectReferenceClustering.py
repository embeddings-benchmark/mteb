from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class HumanMicrobiomeProjectReferenceClusteringP2P(AbsTaskClusteringFast):
    max_document_to_embed = 27 # aligned with the train size
    max_fraction_of_documents_to_embed = None
    max_depth = 5

    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectReferenceClusteringP2P",
        description="A P2P clustering task on the Human Microbiome Project Reference dataset.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanMicrobiomeProjectReference",
        dataset={
            "path": "metagene-ai/HumanMicrobiomeProjectReference",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="v_measure",
        date=("2009-10-09", "2012-11-22"),
        domains=["Medical"],
        task_subtypes=None,
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{liu2025metagene,
            title={METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring},
            author={Liu, Ollie and Jaghouar, Sami and Hagemann, Johannes and Wang, Shangshang and Wiemels, Jason and Kaufman, Jeff and Neiswanger, Willie},
            journal={arXiv preprint arXiv:2501.02045},
            year={2025}
        }
        """,
        prompt="Identify virus taxonomy based on metagenomic sequences",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        from transformers.trainer_utils import set_seed
        set_seed(42)

        import datasets
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])

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
        self.dataset = self.dataset.rename_columns({"sequence": "sentences", "taxonomy": "labels"})
        def remove_trivial_level(data):
            data["labels"] = data["labels"][1:]
            return data
        self.dataset = self.dataset.map(remove_trivial_level)


class HumanMicrobiomeProjectReferenceClusteringS2SAlign(AbsTaskClusteringFast):
    max_document_to_embed = 2198 # When named with "align", this max_document_to_embed is set as the number of train data in the train split.
    max_fraction_of_documents_to_embed = None
    max_depth = 5

    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectReferenceClusteringS2SAlign",
        description="A S2S clustering task on the Human Microbiome Project Reference dataset.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanMicrobiomeProjectReference",
        dataset={
            "path": "metagene-ai/HumanMicrobiomeProjectReference",
            "revision": "main",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="v_measure",
        date=("2009-10-09", "2012-11-22"),
        domains=["Medical"],
        task_subtypes=None,
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{liu2025metagene,
            title={METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring},
            author={Liu, Ollie and Jaghouar, Sami and Hagemann, Johannes and Wang, Shangshang and Wiemels, Jason and Kaufman, Jeff and Neiswanger, Willie},
            journal={arXiv preprint arXiv:2501.02045},
            year={2025}
        }
        """,
        prompt="Identify virus taxonomy based on metagenomic sequences",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        from transformers.trainer_utils import set_seed
        set_seed(42)

        import datasets
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])

        self.dataset_transform()

        full_train_dataset = self.dataset['train']
        test_size = 0.995
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
        self.dataset = self.dataset.rename_columns({"sequence": "sentences", "taxonomy": "labels"})
        def remove_trivial_level(data):
            data["labels"] = data["labels"][1:]
            return data
        self.dataset = self.dataset.map(remove_trivial_level)

        dataset = self.dataset["train"]

        dataset_chunk = []
        label_chunk = []

        import random
        for sentence, label in zip(dataset["sentences"], dataset["labels"]):
            chunks = []
            target_size = 200
            target_overlap = 50

            size_std = target_size * 0.2
            overlap_std = target_overlap * 0.2

            current_pos = 0
            while current_pos < len(sentence):
                chunk_size = int(random.gauss(target_size, size_std))

                if current_pos + chunk_size <= len(sentence):
                    chunk = sentence[current_pos:current_pos + chunk_size]
                    chunks.append(chunk)

                overlap = int(random.gauss(target_overlap, overlap_std))
                current_pos += chunk_size - overlap

            split_idx = int(len(chunks) * 0.8)

            dataset_chunk.extend(chunks)
            label_chunk.extend([label] * len(chunks))

        import datasets
        self.dataset["train"] = datasets.Dataset.from_dict({
            "sentences": dataset_chunk,
            "labels": label_chunk
        })


class HumanMicrobiomeProjectReferenceClusteringS2SSmall(AbsTaskClusteringFast):
    max_document_to_embed = 27 # When named with "small", the max_document_to_embed is the same as the max_document_to_embed of its P2P counterpart.
    max_fraction_of_documents_to_embed = None
    max_depth = 5

    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectReferenceClusteringS2SSmall",
        description="A S2S clustering task on the Human Microbiome Project Reference dataset.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanMicrobiomeProjectReference",
        dataset={
            "path": "metagene-ai/HumanMicrobiomeProjectReference",
            "revision": "main",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="v_measure",
        date=("2009-10-09", "2012-11-22"),
        domains=["Medical"],
        task_subtypes=None,
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{liu2025metagene,
            title={METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring},
            author={Liu, Ollie and Jaghouar, Sami and Hagemann, Johannes and Wang, Shangshang and Wiemels, Jason and Kaufman, Jeff and Neiswanger, Willie},
            journal={arXiv preprint arXiv:2501.02045},
            year={2025}
        }
        """,
        prompt="Identify virus taxonomy based on metagenomic sequences",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        from transformers.trainer_utils import set_seed
        set_seed(42)

        import datasets
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])

        self.dataset_transform()

        full_train_dataset = self.dataset['train']
        test_size = 0.995
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
        self.dataset = self.dataset.rename_columns({"sequence": "sentences", "taxonomy": "labels"})
        def remove_trivial_level(data):
            data["labels"] = data["labels"][1:]
            return data
        self.dataset = self.dataset.map(remove_trivial_level)

        dataset = self.dataset["train"]

        dataset_chunk = []
        label_chunk = []

        import random
        for sentence, label in zip(dataset["sentences"], dataset["labels"]):
            chunks = []
            target_size = 200
            target_overlap = 50

            size_std = target_size * 0.2
            overlap_std = target_overlap * 0.2

            current_pos = 0
            while current_pos < len(sentence):
                chunk_size = int(random.gauss(target_size, size_std))

                if current_pos + chunk_size <= len(sentence):
                    chunk = sentence[current_pos:current_pos + chunk_size]
                    chunks.append(chunk)

                overlap = int(random.gauss(target_overlap, overlap_std))
                current_pos += chunk_size - overlap

            split_idx = int(len(chunks) * 0.8)

            dataset_chunk.extend(chunks)
            label_chunk.extend([label] * len(chunks))

        import datasets
        self.dataset["train"] = datasets.Dataset.from_dict({
            "sentences": dataset_chunk,
            "labels": label_chunk
        })


class HumanMicrobiomeProjectReferenceClusteringS2STiny(AbsTaskClusteringFast):
    max_document_to_embed = 3 # When named with "tiny", the max_document_to_embed is set as "3" to simulate the "3-mer" in DNA sequence processing.
    max_fraction_of_documents_to_embed = None
    max_depth = 5

    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectReferenceClusteringS2STiny",
        description="A S2S clustering task on the Human Microbiome Project Reference dataset.",
        reference="https://huggingface.co/datasets/metagene-ai/HumanMicrobiomeProjectReference",
        dataset={
            "path": "metagene-ai/HumanMicrobiomeProjectReference",
            "revision": "main",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="v_measure",
        date=("2009-10-09", "2012-11-22"),
        domains=["Medical"],
        task_subtypes=None,
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{liu2025metagene,
            title={METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring},
            author={Liu, Ollie and Jaghouar, Sami and Hagemann, Johannes and Wang, Shangshang and Wiemels, Jason and Kaufman, Jeff and Neiswanger, Willie},
            journal={arXiv preprint arXiv:2501.02045},
            year={2025}
        }
        """,
        prompt="Identify virus taxonomy based on metagenomic sequences",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        from transformers.trainer_utils import set_seed
        set_seed(42)

        import datasets
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])

        self.dataset_transform()

        full_train_dataset = self.dataset['train']
        test_size = 0.995
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
        self.dataset = self.dataset.rename_columns({"sequence": "sentences", "taxonomy": "labels"})
        def remove_trivial_level(data):
            data["labels"] = data["labels"][1:]
            return data
        self.dataset = self.dataset.map(remove_trivial_level)

        dataset = self.dataset["train"]

        dataset_chunk = []
        label_chunk = []

        import random
        for sentence, label in zip(dataset["sentences"], dataset["labels"]):
            chunks = []
            target_size = 200
            target_overlap = 50

            size_std = target_size * 0.2
            overlap_std = target_overlap * 0.2

            current_pos = 0
            while current_pos < len(sentence):
                chunk_size = int(random.gauss(target_size, size_std))

                if current_pos + chunk_size <= len(sentence):
                    chunk = sentence[current_pos:current_pos + chunk_size]
                    chunks.append(chunk)

                overlap = int(random.gauss(target_overlap, overlap_std))
                current_pos += chunk_size - overlap

            split_idx = int(len(chunks) * 0.8)

            dataset_chunk.extend(chunks)
            label_chunk.extend([label] * len(chunks))

        import datasets
        self.dataset["train"] = datasets.Dataset.from_dict({
            "sentences": dataset_chunk,
            "labels": label_chunk
        })