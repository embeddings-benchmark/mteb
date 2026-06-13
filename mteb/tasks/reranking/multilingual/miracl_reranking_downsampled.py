from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "dev"
_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "yo": ["yor-Latn"],
    "zh": ["zho-Hans"],
}

_CITATION = r"""@article{10.1162/tacl_a_00595,
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  doi = {10.1162/tacl_a_00595},
  issn = {2307-387X},
  journal = {Transactions of the Association for Computational Linguistics},
  month = {09},
  pages = {1114-1131},
  title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
  volume = {11},
  year = {2023},
}"""


class MIRACLRerankingDownsampled(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRerankingDownsampled",
        description="A downsampled version of MIRACLReranking with top-70 candidates per query (truncated from the original 100). Reduces corpus size by ~30% while maintaining ranking quality within ±2% NDCG@10 of the full version.",
        reference="https://project-miracl.github.io/",
        dataset={
            "path": "embedding-benchmark/MIRACLRerankingDownsampled",
            "revision": "1c95f69fca65ecb31102be9746c0fcd2b0ffa664",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-06-01", "2023-01-30"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["MIRACLReranking"],
    )
