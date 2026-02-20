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

_common_metadata = dict(
    reference="http://miracl.ai",
    type="Retrieval",
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
    bibtex_citation=r"""
@article{10.1162/tacl_a_00595,
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  doi = {10.1162/tacl_a_00595},
  eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
  issn = {2307-387X},
  journal = {Transactions of the Association for Computational Linguistics},
  month = {09},
  pages = {1114-1131},
  title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
  url = {https://doi.org/10.1162/tacl\_a\_00595},
  volume = {11},
  year = {2023},
}
""",
)


class MIRACLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrieval",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        dataset={
            "path": "mteb/MIRACLRetrieval",
            "revision": "9c09abc13478308c27598f350e31d8f06b9b5481",
        },
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        **_common_metadata,
    )


class MIRACLRetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrievalHardNegatives",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        dataset={
            "path": "mteb/MIRACLRetrievalHardNegatives",
            "revision": "d7d94fa4b946cec4a27c84653aa0cf6b33f74a3c",
        },
        adapted_from=["MIRACLRetrieval"],
        **_common_metadata,
    )


class MIRACLRetrievalHardNegativesV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrievalHardNegatives.v2",
        description=(
            "MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval "
            "dataset that focuses on search across 18 different languages. The hard negative version has been "
            "created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        dataset={
            "path": "mteb/MIRACLRetrievalHardNegatives",
            "revision": "d7d94fa4b946cec4a27c84653aa0cf6b33f74a3c",
        },
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question",
        },
        adapted_from=["MIRACLRetrieval"],
        **_common_metadata,
    )
