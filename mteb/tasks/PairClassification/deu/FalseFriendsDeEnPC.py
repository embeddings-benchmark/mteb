from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


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
        category="s2s",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ap",
        date=("2023-08-01", "2023-09-01"),
        form=["written"],
        domains=[],
        task_subtypes=["False Friends"],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @misc{Chibb_2022, 
        title="German-English False Friends in Multilingual Transformer Models: An Evaluation on Robustness and Word-to-Word Fine-Tuning",
        author="Chibb, Aaron",
        year="2022", 
        month="Sep"
        abstract="This paper explores the robustness of multilingual language models against false friends. False friends are words that sound or are written the same in two different languages but have different meaning. Generally, it is argued that multilingual models, such as XLM-RoBERTA, can outperform monolingual models in most tasks on conventional datasets. However, false friends are not considered in these tests. In this paper, experiments with a false friends dataset show that multilingual models are not robust against false friends; they have problems creating monolingual representations and differentiating between meanings of similarly written words in different languages. An attempt of word-based finetuning multilingual models on false friends pairs is promising, however the results do not generally solve the presented problem and still, monolingual models are more robust against false friends."
        }
        """,
        n_samples={"test": 1524},
        avg_character_length={"test": 40.3},
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]

            _dataset[split] = [
                {
                    "sent1": hf_dataset["sent1"],
                    "sent2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
