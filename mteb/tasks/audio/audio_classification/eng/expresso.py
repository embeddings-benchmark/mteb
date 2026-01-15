from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ExpressoReadEmotionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ExpressoRead",
        description="Multiclass expressive speech style classification. This is a stratfied and downsampled version of the original dataset that contains 40 hours of speech. The original dataset has two subsets - read speech and conversational speech, each having their own set of style labels. This task only includes the read speech subset.",
        reference="https://speechbot.github.io/expresso/",
        dataset={
            "path": "mteb/expresso-read-mini",
            "revision": "cf3bf160c3b18b5e0aa6607d8a724f5e397c2801",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-13", "2025-01-13"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Emotion classification"],
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


class ExpressoConvEmotionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ExpressoConv",
        description="Multiclass expressive speech style classification. This is a stratfied and downsampled version of the original dataset that contains 40 hours of speech. The original dataset has two subsets - read speech and conversational speech, each having their own set of style labels. This task only includes the conversational speech subset.",
        reference="https://speechbot.github.io/expresso/",
        dataset={
            "path": "mteb/expresso-conv-mini",
            "revision": "fcb3c426f790493d4df74a9cf4d3109641e4de26",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-13", "2025-01-13"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Emotion classification"],
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
