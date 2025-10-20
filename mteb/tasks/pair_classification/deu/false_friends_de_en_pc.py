from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FalseFriendsDeEnPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FalseFriendsGermanEnglish",
        description="A dataset to identify False Friends / false cognates between English and German. A generally challenging task for multilingual models.",
        reference="https://drive.google.com/file/d/1jgq0nBnV-UiYNxbKNrrr2gxDEHm-DMKH/view?usp=share_link",
        dataset={
            "path": "aari1995/false_friends_de_en_mteb",
            "revision": "15d6c030d3336cbb09de97b2cefc46db93262d40",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="max_ap",
        date=("2023-08-01", "2023-09-01"),
        domains=["Written"],
        task_subtypes=["False Friends"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{Chibb_2022,
  author = {Chibb, Aaron},
  month = {Sep},
  title = {{German-English False Friends in Multilingual Transformer Models: An Evaluation on Robustness and Word-to-Word Fine-Tuning}},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]

            _dataset[split] = [
                {
                    "sentence1": hf_dataset["sent1"],
                    "sentence2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
