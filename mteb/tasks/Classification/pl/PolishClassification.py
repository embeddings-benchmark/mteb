from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class CbdClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CBD",
        description="Polish Tweets annotated for cyberbullying detection.",
        reference="http://2019.poleval.pl/files/poleval2019.pdf",
        hf_hub_name="PL-MTEB/cbd",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="accuracy",
        revision="59d12749a3c91a186063c7d729ec392fda94681c",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


class PolEmo2InClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-IN",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.",
        reference="https://aclanthology.org/K19-1092.pdf",
        hf_hub_name="PL-MTEB/polemo2_in",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="accuracy",
        revision="9e9b1f8ef51616073f47f306f7f47dd91663f86a",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


class PolEmo2OutClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-OUT",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
        "school) reviews using models train on reviews from medicine and hotels domains.",
        reference="https://aclanthology.org/K19-1092.pdf",
        hf_hub_name="PL-MTEB/polemo2_out",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="accuracy",
        revision="c99d599f0a6ab9b85b065da6f9d94f9cf731679f",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


class AllegroReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AllegroReviews",
        description="A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.",
        reference="https://aclanthology.org/2020.acl-main.111.pdf",
        hf_hub_name="PL-MTEB/allegro-reviews",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="accuracy",
        revision="477b8bd4448b5ef8ed01ba82bf9ff67f6e109207",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


class PacClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PAC",
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2211.13112.pdf",
        hf_hub_name="laugustyniak/abusive-clauses-pl",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="accuracy",
        revision="8a04d940a42cd40658986fdd8e3da561533a3646",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
