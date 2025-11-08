from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


def combine_dialogs(row: dict) -> dict:
    row["dialog"] = "\n".join(row["dialog"])
    return row


class DailyDialogClassificationAct(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DailyDialogClassificationAct",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog",
            "revision": "12f30df52a776875c2e334d9d051fa67afd15756",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2017dailydialogmanuallylabelledmultiturn,
      title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset}, 
      author={Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
      year={2017},
      eprint={1710.03957},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1710.03957}, 
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"act_label": "label", "dialog": "text"}
        )


class DailyDialogClassificationEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DailyDialogClassificationEmotion",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog",
            "revision": "12f30df52a776875c2e334d9d051fa67afd15756",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2017dailydialogmanuallylabelledmultiturn,
      title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset}, 
      author={Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
      year={2017},
      eprint={1710.03957},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1710.03957}, 
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"emotion_label": "label", "dialog": "text"}
        )
