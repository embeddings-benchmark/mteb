from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EmotionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EmotionClassification",
        description="Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
        reference="https://www.aclweb.org/anthology/D18-1404",
        dataset={
            "path": "mteb/emotion",
            "revision": "4f58c6b202a23cf9a4da393831edf4f9183cad37",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2018-12-31",
        ),  # Estimated range for the collection of Twitter messages
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{saravia-etal-2018-carer,
  address = {Brussels, Belgium},
  author = {Saravia, Elvis  and
Liu, Hsien-Chi Toby  and
Huang, Yen-Hao  and
Wu, Junlin  and
Chen, Yi-Shin},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1404},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {3687--3697},
  publisher = {Association for Computational Linguistics},
  title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
  url = {https://aclanthology.org/D18-1404},
  year = {2018},
}
""",
        prompt="Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
        superseded_by="EmotionClassification.v2",
    )

    samples_per_label = 16


class EmotionClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EmotionClassification.v2",
        description="Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://www.aclweb.org/anthology/D18-1404",
        dataset={
            "path": "mteb/emotion",
            "revision": "13535ec7ed83ac3920c40db3c3fd4133af55cc06",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-01-01",
            "2018-12-31",
        ),  # Estimated range for the collection of Twitter messages
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{saravia-etal-2018-carer,
  address = {Brussels, Belgium},
  author = {Saravia, Elvis  and
Liu, Hsien-Chi Toby  and
Huang, Yen-Hao  and
Wu, Junlin  and
Chen, Yi-Shin},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1404},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {3687--3697},
  publisher = {Association for Computational Linguistics},
  title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
  url = {https://aclanthology.org/D18-1404},
  year = {2018},
}
""",
        prompt="Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
        adapted_from=["EmotionClassification"],
    )

    samples_per_label = 16
