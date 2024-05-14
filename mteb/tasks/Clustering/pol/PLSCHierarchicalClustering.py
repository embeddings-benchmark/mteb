from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2048


class PLSCHierarchicalClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PLSCHierarchicalClusteringP2P",
        description="Clustering of Polish article titles+abstracts from Library of Science "
        "(https://bibliotekanauki.pl/), either on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "rafalposwiata/plsc",
            "revision": "b8b7667d452b5f3ac16fd77b84823d00599e2656",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": N_SAMPLES},
        avg_character_length={"test": 1023.21},
    )

    def dataset_transform(self):
        self.dataset = self.dataset["train"].train_test_split(
            seed=self.seed, test_size=N_SAMPLES
        )
        self.dataset = self.dataset.rename_columns(
            {"abstract": "sentences", "disciplines": "labels"}
        )


class PLSCHierarchicalClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PLSCHierarchicalClusteringS2S",
        description="Clustering of Polish article titles+abstracts from Library of Science "
        "(https://bibliotekanauki.pl/), either on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "rafalposwiata/plsc",
            "revision": "b8b7667d452b5f3ac16fd77b84823d00599e2656",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": N_SAMPLES},
        avg_character_length={"test": 1023.21},
    )

    def dataset_transform(self):
        self.dataset = self.dataset["train"].train_test_split(
            seed=self.seed, test_size=N_SAMPLES
        )
        self.dataset = self.dataset.rename_columns(
            {"title": "sentences", "disciplines": "labels"}
        )
