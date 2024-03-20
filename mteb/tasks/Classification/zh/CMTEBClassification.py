from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class TNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TNews",
        description="Short Text Classification for News",
        reference="https://www.cluebenchmarks.com/introduce.html",
        hf_hub_name="C-MTEB/TNews-classification",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="317f262bf1e6126357bbe89e875451e4b0938fe4",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32
        return metadata_dict


class IFlyTek(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IFlyTek",
        description="Long Text classification for the description of Apps",
        reference="https://www.cluebenchmarks.com/introduce.html",
        hf_hub_name="C-MTEB/IFlyTek-classification",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="421605374b29664c5fc098418fe20ada9bd55f8a",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32
        metadata_dict["n_experiments"] = 5
        return metadata_dict


class MultilingualSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualSentiment",
        description="A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative",
        reference="https://github.com/tyqiangz/multilingual-sentiment-datasets",
        hf_hub_name="C-MTEB/MultilingualSentiment-classification",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="46958b007a63fdbf239b7672c25d0bea67b5ea1a",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32
        return metadata_dict


class JDReview(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JDReview",
        description="review for iphone",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="C-MTEB/JDReview-classification",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="b7c64bd89eb87f8ded463478346f76731f07bf8b",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32
        return metadata_dict


class OnlineShopping(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OnlineShopping",
        description="Sentiment Analysis of User Reviews on Online Shopping Websites",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="C-MTEB/OnlineShopping-classification",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="e610f2ebd179a8fda30ae534c3878750a96db120",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32
        return metadata_dict


class Waimai(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Waimai",
        description="Sentiment Analysis of user reviews on takeaway platforms",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="C-MTEB/waimai-classification",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="accuracy",
        revision="339287def212450dcaa9df8c22bf93e9980c7023",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["samples_per_label"] = 32

        return metadata_dict
