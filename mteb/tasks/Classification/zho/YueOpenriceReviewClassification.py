from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class YueOpenriceReviewClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YueOpenriceReviewClassification",
        description="A Cantonese dataset for review classification",
        reference="https://github.com/Christainx/Dataset_Cantonese_Openrice",
        dataset={
            "path": "izhx/yue-openrice-review",
            "revision": "1300d045cf983bac23faadf3aa12a619624769da",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["yue-Hant"],
        main_score="accuracy",
        date=("2019-01-01", "2019-05-01"),
        form=["spoken"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{xiang2019sentiment,
  title={Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
  author={Xiang, Rong and Jiao, Ying and Lu, Qin},
  booktitle={Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
  pages={1--9},
  year={2019},
  organization={KDD WISDOM}
}""",
        n_samples={"test": 6161},
        avg_character_length={"test": 173.0},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
