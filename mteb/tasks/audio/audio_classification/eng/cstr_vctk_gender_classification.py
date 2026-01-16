from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CSTRVCTKGenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSTRVCTKGender",
        description="Gender classification from CSTR-VCTK dataset. This is a stratified and downsampled version of the original dataset. The dataset was recorded with 2 different microphones, and this mini version uniformly samples data from the 2 microphone types.",
        reference="https://datashare.ed.ac.uk/handle/10283/3443",
        dataset={
            "path": "mteb/cstr-vctk-gender-mini",
            "revision": "fe8fbc6d596d805316883bb0bce0b534da008123",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-11-15", "2025-11-15"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Yamagishi2019CSTRVC,
  title = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)},
  author = {Junichi Yamagishi and Christophe Veaux and Kirsten MacDonald},
  year = {2019},
  url = {https://api.semanticscholar.org/CorpusID:213060286}
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "gender_id"

    def dataset_transform(self):
        # Define label mapping
        label2id = {"F": 0, "M": 1}

        # Apply transformation to all dataset splits
        for split in self.dataset:
            # Define transform function to add numeric labels
            def add_gender_id(example):
                example["gender_id"] = label2id[example["gender"]]
                return example

            self.dataset[split] = self.dataset[split].map(add_gender_id)
