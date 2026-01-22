from mteb.abstasks import AbsTaskZeroShotClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SpeechCommandsZeroshotClassificationV01(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="SpeechCommandsZeroshotv0.01",
        description="Sound Classification/Keyword Spotting Dataset. This is a set of one-second audio clips containing a single spoken English word or background noise. These words are from a small set of commands such as 'yes', 'no', and 'stop' spoken by various speakers. With a total of 10 labels/commands for keyword spotting and a total of 30 labels for other auxiliary tasks",
        reference="https://huggingface.co/datasets/google/speech_commands",
        dataset={
            "path": "mteb/SpeechCommandsZeroshotv0.01",
            "revision": "200a38f4d145460758244d6613cb08dffe70fa8a",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-07-07", "2018-07-13"),
        domains=["Spoken"],
        task_subtypes=["Keyword Spotting"],
        license="cc-by-4.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-1804-03209,
  author = {Pete Warden},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1804-03209.bib},
  eprint = {1804.03209},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:48:32 +0200},
  title = {Speech Commands: {A} Dataset for Limited-Vocabulary Speech Recognition},
  url = {http://arxiv.org/abs/1804.03209},
  volume = {abs/1804.03209},
  year = {2018},
}
""",
    )
    input_column_name: str = "audio"
    label_column_name: str = "label"

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "Yes",
            "No",
            "Up",
            "Down",
            "Left",
            "Right",
            "On",
            "Off",
            "Stop",
            "Go",
            # Dataset has 30 labels, but only first 10 are used for zeroshot classification since they are considered as commands, others are considered as auxiliary labels for v1.1
        ]


class SpeechCommandsZeroshotClassificationv02(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="SpeechCommandsZeroshotv0.02",
        description="Sound Classification/Keyword Spotting Dataset. This is a set of one-second audio clips containing a single spoken English word or background noise. These words are from a small set of commands such as 'yes', 'no', and 'stop' spoken by various speakers. With a total of 10 labels/commands for keyword spotting and a total of 30 labels for other auxiliary tasks",
        reference="https://huggingface.co/datasets/google/speech_commands",
        dataset={
            "path": "mteb/SpeechCommandsZeroshotv0.02",
            "revision": "d5491b288ab76356a5f3cbc634f1dc38e23da2b8",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-07-07", "2018-07-13"),
        domains=["Spoken"],
        task_subtypes=["Keyword Spotting"],
        license="cc-by-4.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-1804-03209,
  author = {Pete Warden},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1804-03209.bib},
  eprint = {1804.03209},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:48:32 +0200},
  title = {Speech Commands: {A} Dataset for Limited-Vocabulary Speech Recognition},
  url = {http://arxiv.org/abs/1804.03209},
  volume = {abs/1804.03209},
  year = {2018},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "Yes",
            "No",
            "Up",
            "Down",
            "Left",
            "Right",
            "On",
            "Off",
            "Stop",
            "Go",
            # Dataset has 30 labels, but only first 10 are used for zeroshot classification since they are considered as commands, others are considered as auxiliary labels for v1.1
        ]
