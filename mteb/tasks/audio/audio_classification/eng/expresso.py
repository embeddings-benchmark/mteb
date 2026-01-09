from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ExpressoExpressiveStyleClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Expresso",
        description="Multiclass expressive speech style classification",
        reference="https://huggingface.co/datasets/ylacombe/expresso",
        dataset={
            "path": "ylacombe/expresso",
            "revision": "9fb79a189698de3255eff48edd2bc0d9e487adc0",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-04-30", "2024-04-30"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Expressive Style Classification"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{nguyen2023expresso,
  author = {Nguyen, Tu Anh and Hsu, Wei-Ning and d'Avirro, Antony and Shi, Bowen and Gat, Itai and Fazel-Zarani, Maryam and Remez, Tal and Copet, Jade and Synnaeve, Gabriel and Hassid, Michael and others},
  booktitle = {INTERSPEECH 2023-24th Annual Conference of the International Speech Communication Association},
  pages = {4823--4827},
  title = {Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
  year = {2023},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "style"

    is_cross_validation: bool = True

    def dataset_transform(self):
        label2id = {
            "confused": 0,
            "sad": 1,
            "essentials": 2,
            "emphasis": 3,
            "default": 4,
            "longform": 5,
            "happy": 6,
            "singing": 7,
            "whisper": 8,
            "enunciated": 9,
            "laughing": 10,
        }

        # Apply transformation to all dataset splits
        for split in self.dataset:
            # Define transform function to add numeric labels
            def add_style_id(example):
                example["style_id"] = label2id[example["style"]]
                return example

            print(f"Converting style labels to numeric IDs for split '{split}'...")
            self.dataset[split] = self.dataset[split].map(add_style_id)
