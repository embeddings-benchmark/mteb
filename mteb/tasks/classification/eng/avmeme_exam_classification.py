from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AVMemeAudioVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AVMemeAudioVideoClassification",
        description="AVMeme Exam is a multimodal, multilingual, and multicultural benchmark of over 1,000 iconic Internet audio-visual memes spanning speech, songs, music, and sound effects. Each meme is paired with a Q&A assessing levels of understanding from surface content to context, emotion, usage, and world knowledge. This classification task predicts the sound category of each meme clip.",
        reference="https://arxiv.org/pdf/2601.17645",
        dataset={
            "path": "mteb/AVMeme-Exam",
            "revision": "7070d1979d9a4943dd49b2e72858eb1e54f6bd5b",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=[
            "bos-Latn",
            "bre-Latn",
            "deu-Latn",
            "eng-Latn",
            "fas-Arab",
            "fin-Latn",
            "fra-Latn",
            "hin-Deva",
            "ita-Latn",
            "jpn-Jpan",
            "kor-Hang",
            "por-Latn",
            "rus-Cyrl",
            "spa-Latn",
            "tel-Telu",
            "zho-Hans",
        ],
        main_score="accuracy",
        date=("2026-01-25", "2026-01-25"),
        domains=["Web", "Entertainment", "Music"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{jiang2026avmeme,
  author = {Jiang, Xilin and Wang, Qiaolin and Wu, Junkai and He, Xiaomin and Xu, Zhongweiyang and Ma, Yinghao and Piao, Minshuo and Yang, Kaiyi and Zheng, Xiuwen and Shimizu, Riki and others},
  journal = {arXiv preprint arXiv:2601.17645},
  title = {AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking},
  year = {2026},
}
""",
        is_beta=True,
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "category"
    is_cross_validation: bool = True
    train_split: str = "test"


class AVMemeVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AVMemeVideoClassification",
        description="AVMeme Exam is a multimodal, multilingual, and multicultural benchmark of over 1,000 iconic Internet audio-visual memes spanning speech, songs, music, and sound effects. Each meme is paired with a Q&A assessing levels of understanding from surface content to context, emotion, usage, and world knowledge. This classification task predicts the sound category of each meme clip.",
        reference="https://arxiv.org/pdf/2601.17645",
        dataset={
            "path": "mteb/AVMeme-Exam",
            "revision": "7070d1979d9a4943dd49b2e72858eb1e54f6bd5b",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=[
            "bos-Latn",
            "bre-Latn",
            "deu-Latn",
            "eng-Latn",
            "fas-Arab",
            "fin-Latn",
            "fra-Latn",
            "hin-Deva",
            "ita-Latn",
            "jpn-Jpan",
            "kor-Hang",
            "por-Latn",
            "rus-Cyrl",
            "spa-Latn",
            "tel-Telu",
            "zho-Hans",
        ],
        main_score="accuracy",
        date=("2026-01-25", "2026-01-25"),
        domains=["Web", "Entertainment", "Music"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{jiang2026avmeme,
  author = {Jiang, Xilin and Wang, Qiaolin and Wu, Junkai and He, Xiaomin and Xu, Zhongweiyang and Ma, Yinghao and Piao, Minshuo and Yang, Kaiyi and Zheng, Xiuwen and Shimizu, Riki and others},
  journal = {arXiv preprint arXiv:2601.17645},
  title = {AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking},
  year = {2026},
}
""",
        is_beta=True,
    )
    input_column_name = "video"
    label_column_name: str = "category"
    is_cross_validation: bool = True
    train_split: str = "test"
