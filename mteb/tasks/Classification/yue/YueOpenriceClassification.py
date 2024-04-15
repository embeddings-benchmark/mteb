from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class YueOpenriceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YueOpenriceClassification",
        description="A Cantonese dataset for sentiment classification on review",
        reference="https://github.com/Christainx/Dataset_Cantonese_Openrice",
        dataset={
            "path": "izhx/yue-openrice-senti",
            "revision": "dcca176e92a9aed684f9c172f3b6c66295d7f04b",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={"yue": ["yue-Hant"]},
        main_score="accuracy",
        date=None,
        form=None,
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{xiang2019sentiment,
  title={Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
  author={Xiang, Rong and Jiao, Ying and Lu, Qin},
  booktitle={Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
  pages={1--9},
  year={2019},
  organization={KDD WISDOM}
}""",
        n_samples={"test": 12322},
        avg_character_length={"test": 173.2},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict
