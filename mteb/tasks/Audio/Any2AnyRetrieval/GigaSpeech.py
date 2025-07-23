from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class GigaSpeechA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="GigaSpeechA2TRetrieval",
        description=(
            "Transcriptions of audio segments drawn from the GigaSpeech "
            "multi-domain English speech corpus, covering audiobooks, podcasts, "
            "and YouTube."
        ),
        reference="https://github.com/SpeechColab/GigaSpeech",
        dataset={
            "path": "mteb/gigaspeech_a2t",
            "revision": "4b41707fe962ab4f948d8826f8b3233929b76237",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{GigaSpeech2021,
  author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
  booktitle = {Proc. Interspeech 2021},
  title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
  year = {2021},
}
""",
    )


class GigaSpeechT2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Text-to-audio retrieval on the GigaSpeech transcription ↔ audio pairs."""

    metadata = TaskMetadata(
        name="GigaSpeechT2ARetrieval",
        description=(
            "Transcriptions of audio segments drawn from the GigaSpeech "
            "multi-domain English speech corpus, covering audiobooks, podcasts, "
            "and YouTube."
        ),
        reference="https://github.com/SpeechColab/GigaSpeech",
        dataset={
            "path": "mteb/gigaspeech_t2a",
            "revision": "38876698eba40ac88433e0ca0d9dff6f35e4d0ff",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{GigaSpeech2021,
  author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
  booktitle = {Proc. Interspeech 2021},
  title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
  year = {2021},
}
""",
    )
