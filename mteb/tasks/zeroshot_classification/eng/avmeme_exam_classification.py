from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{jiang2026avmeme,
  author = {Jiang, Xilin and Wang, Qiaolin and Wu, Junkai and He, Xiaomin and Xu, Zhongweiyang and Ma, Yinghao and Piao, Minshuo and Yang, Kaiyi and Zheng, Xiuwen and Shimizu, Riki and others},
  journal = {arXiv preprint arXiv:2601.17645},
  title = {AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking},
  year = {2026},
}
"""


class AVMemeAudioVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeAudioVideoZeroShot",
        description="AVMeme Exam is a multimodal, multilingual, and multicultural benchmark of over 1,000 iconic Internet audio-visual memes spanning speech, songs, music, and sound effects. Each meme is paired with a Q&A assessing levels of understanding from surface content to context, emotion, usage, and world knowledge. This zero-shot classification task predicts the sound category of each meme clip.",
        reference="https://arxiv.org/pdf/2601.17645",
        dataset={
            "path": "mteb/AVMemeAudioVideoZeroShot",
            "revision": "b5e76b9f1a358521e6d99eb9c34ed58fb4da9c12",
        },
        type="VideoZeroshotClassification",
        category="va2t",
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
        domains=["Entertainment", "Social", "Web"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class AVMemeVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeVideoZeroShot",
        description="AVMeme Exam is a multimodal, multilingual, and multicultural benchmark of over 1,000 iconic Internet audio-visual memes spanning speech, songs, music, and sound effects. Each meme is paired with a Q&A assessing levels of understanding from surface content to context, emotion, usage, and world knowledge. This zero-shot classification task predicts the sound category of each meme clip.",
        reference="https://arxiv.org/pdf/2601.17645",
        dataset={
            "path": "mteb/AVMemeVideoZeroShot",
            "revision": "7c4c98c10779d155126fee259acfd85dbd11318a",
        },
        type="VideoZeroshotClassification",
        category="v2t",
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
        domains=["Entertainment", "Social", "Web"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
