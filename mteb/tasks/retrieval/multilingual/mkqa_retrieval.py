from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "ar": ["ara-Arab"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "he": ["heb-Hebr"],
    "hu": ["hun-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "km": ["khm-Khmr"],
    "ms": ["msa-Latn"],
    "nl": ["nld-Latn"],
    "no": ["nor-Latn", "nno-Latn", "nob-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "sv": ["swe-Latn"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "vi": ["vie-Latn"],
    "zh_cn": ["zho-Hans"],
    "zh_hk": ["zho-Hant"],
    "zh_tw": ["zho-Hant"],
}


class MKQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MKQARetrieval",
        description="Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset. For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.",
        reference="https://github.com/apple/ml-mkqa",
        dataset={
            "path": "mteb/MKQARetrieval",
            "revision": "3e069e7a30079d214a859741f5a0de75cf878867",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        date=("2020-01-01", "2020-12-31"),
        eval_splits=["train"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        domains=["Written"],
        task_subtypes=["Question answering"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{mkqa,
  author = {Shayne Longpre and Yi Lu and Joachim Daiber},
  title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
  url = {https://arxiv.org/pdf/2007.15207.pdf},
  year = {2020},
}
""",
    )
