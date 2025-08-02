from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakHateSpeechClassification(AbsTaskClassification):
    superseded_by = "SlovakHateSpeechClassification.v2"
    metadata = TaskMetadata(
        name="SlovakHateSpeechClassification",
        description="The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.",
        reference="https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak",
        dataset={
            "path": "TUKE-KEMT/hate_speech_slovak",
            "revision": "f9301b9937128c9c0b636fa6da203aeb046479f4",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2024-05-25", "2024-06-06"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )


class SlovakHateSpeechClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakHateSpeechClassification.v2",
        description="""The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak",
        dataset={
            "path": "mteb/slovak_hate_speech",
            "revision": "691fe861df0ffa25066cbf6da8e64ebd296af6ab",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2024-05-25", "2024-06-06"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["SlovakHateSpeechClassification"],
    )
