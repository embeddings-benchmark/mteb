from datasets import Sequence, Value
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BirdSetMultilabelClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="BirdSet",
        description="BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics",
        reference="https://huggingface.co/datasets/DBD-research-group/BirdSet",
        dataset={
            "path": "mteb/BirdSet",
            "name": "HSN",
            "revision": "bdaa5020a8dc594a9a1d3b344e6ca9dbfaa33c74",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test_5s"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),  # Competition year
        domains=["Spoken", "Speech", "Bioacoustics"],
        task_subtypes=["Species Classification"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{rauch2024birdsetlargescaledatasetaudio,
  archiveprefix = {arXiv},
  author = {Lukas Rauch and Raphael Schwinger and Moritz Wirth and René Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
  eprint = {2403.10380},
  primaryclass = {cs.SD},
  title = {BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics},
  url = {https://arxiv.org/abs/2403.10380},
  year = {2024},
}
""",
    )

    evaluator_model = MultiOutputClassifier(estimator=LogisticRegression())
    input_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 21

    def dataset_transform(self):
        """Rename ebird_code_multilabel → labels and turn IDs → bird names."""
        if "ebird_code_multilabel" in self.dataset.column_names[self.eval_splits[0]]:
            self.dataset = self.dataset.rename_column(
                original_column_name="ebird_code_multilabel",
                new_column_name=self.label_column_name,  # "labels"
            )

        id2name = {
            0: "Gray-crowned Rosy-Finch",
            1: "White-crowned Sparrow",
            2: "American Pipit",
            3: "Spotted Sandpiper",
            4: "Rock Wren",
            5: "Brewer's Blackbird",
            6: "Dark-eyed Junco",
            7: "Fox Sparrow",
            8: "Clark's Nutcracker",
            9: "Mountain Bluebird",
            10: "Cassin's Finch",
            11: "Mallard",
            12: "Hermit Thrush",
            13: "American Robin",
            14: "Yellow-rumped Warbler",
            15: "Yellow Warbler",
            16: "Dusky Flycatcher",
            17: "Mountain Chickadee",
            18: "Orange-crowned Warbler",
            19: "Warbling Vireo",
            20: "Northern Flicker",
        }

        def ids_to_names(example):
            """Convert a list of ints to a list of bird-name strings."""
            labels = example[self.label_column_name]
            if labels is None:
                return {self.label_column_name: []}
            return {self.label_column_name: [id2name[i] for i in labels]}

        # Cast to int64 to remove ClassLabel schema before mapping to strings
        self.dataset = self.dataset.cast_column(
            self.label_column_name, Sequence(Value("int64"))
        )

        self.dataset = self.dataset.map(
            ids_to_names,
            num_proc=4,
        )

        # Filter out empty labels
        self.dataset = self.dataset.filter(
            lambda x: x[self.label_column_name] is not None
            and len(x[self.label_column_name]) > 0
        )

        # Only subsample splits that are larger than n_samples to avoid division by zero
        n_samples = 2048
        splits_to_subsample = [
            split for split in self.eval_splits if len(self.dataset[split]) > n_samples
        ]

        if splits_to_subsample:
            self.dataset = self.stratified_subsampling(
                self.dataset,
                seed=self.seed,
                splits=splits_to_subsample,
                label=self.label_column_name,
                n_samples=n_samples,
            )
