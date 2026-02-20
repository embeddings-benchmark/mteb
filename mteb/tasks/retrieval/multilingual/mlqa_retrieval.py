from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "mlqa.ar.ar": ["ara-Arab", "ara-Arab"],
    "mlqa.ar.de": ["ara-Arab", "deu-Latn"],
    "mlqa.ar.en": ["ara-Arab", "eng-Latn"],
    "mlqa.ar.es": ["ara-Arab", "spa-Latn"],
    "mlqa.ar.hi": ["ara-Arab", "hin-Deva"],
    "mlqa.ar.vi": ["ara-Arab", "vie-Latn"],
    "mlqa.ar.zh": ["ara-Arab", "zho-Hans"],
    "mlqa.de.ar": ["deu-Latn", "ara-Arab"],
    "mlqa.de.de": ["deu-Latn", "deu-Latn"],
    "mlqa.de.en": ["deu-Latn", "eng-Latn"],
    "mlqa.de.es": ["deu-Latn", "spa-Latn"],
    "mlqa.de.hi": ["deu-Latn", "hin-Deva"],
    "mlqa.de.vi": ["deu-Latn", "vie-Latn"],
    "mlqa.de.zh": ["deu-Latn", "zho-Hans"],
    "mlqa.en.ar": ["eng-Latn", "ara-Arab"],
    "mlqa.en.de": ["eng-Latn", "deu-Latn"],
    "mlqa.en.en": ["eng-Latn", "eng-Latn"],
    "mlqa.en.es": ["eng-Latn", "spa-Latn"],
    "mlqa.en.hi": ["eng-Latn", "hin-Deva"],
    "mlqa.en.vi": ["eng-Latn", "vie-Latn"],
    "mlqa.en.zh": ["eng-Latn", "zho-Hans"],
    "mlqa.es.ar": ["spa-Latn", "ara-Arab"],
    "mlqa.es.de": ["spa-Latn", "deu-Latn"],
    "mlqa.es.en": ["spa-Latn", "eng-Latn"],
    "mlqa.es.es": ["spa-Latn", "spa-Latn"],
    "mlqa.es.hi": ["spa-Latn", "hin-Deva"],
    "mlqa.es.vi": ["spa-Latn", "vie-Latn"],
    "mlqa.es.zh": ["spa-Latn", "zho-Hans"],
    "mlqa.hi.ar": ["hin-Deva", "ara-Arab"],
    "mlqa.hi.de": ["hin-Deva", "deu-Latn"],
    "mlqa.hi.en": ["hin-Deva", "eng-Latn"],
    "mlqa.hi.es": ["hin-Deva", "spa-Latn"],
    "mlqa.hi.hi": ["hin-Deva", "hin-Deva"],
    "mlqa.hi.vi": ["hin-Deva", "vie-Latn"],
    "mlqa.hi.zh": ["hin-Deva", "zho-Hans"],
    "mlqa.vi.ar": ["vie-Latn", "ara-Arab"],
    "mlqa.vi.de": ["vie-Latn", "deu-Latn"],
    "mlqa.vi.en": ["vie-Latn", "eng-Latn"],
    "mlqa.vi.es": ["vie-Latn", "spa-Latn"],
    "mlqa.vi.hi": ["vie-Latn", "hin-Deva"],
    "mlqa.vi.vi": ["vie-Latn", "vie-Latn"],
    "mlqa.vi.zh": ["vie-Latn", "zho-Hans"],
    "mlqa.zh.ar": ["zho-Hans", "ara-Arab"],
    "mlqa.zh.de": ["zho-Hans", "deu-Latn"],
    "mlqa.zh.en": ["zho-Hans", "eng-Latn"],
    "mlqa.zh.es": ["zho-Hans", "spa-Latn"],
    "mlqa.zh.hi": ["zho-Hans", "hin-Deva"],
    "mlqa.zh.vi": ["zho-Hans", "vie-Latn"],
    "mlqa.zh.zh": ["zho-Hans", "zho-Hans"],
}


def _build_lang_pair(langs: list[str]) -> str:
    """Builds a language pair separated by a dash.
    e.g., ['eng-Latn', 'deu-Latn'] -> 'eng-deu'.
    """
    return langs[0].split("-")[0] + "-" + langs[1].split("-")[0]


def extend_lang_pairs() -> dict[str, list[str]]:
    eval_langs = {}
    for langs in _LANGUAGES.values():
        lang_pair = _build_lang_pair(langs)
        eval_langs[lang_pair] = langs
    return eval_langs


_EVAL_LANGS = extend_lang_pairs()


class MLQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MLQARetrieval",
        description="MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance. MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between 4 different languages on average.",
        reference="https://huggingface.co/datasets/mlqa",
        dataset={
            "path": "mteb/MLQARetrieval",
            "revision": "5bef8b6e2601af974fb1a1cca03fd702229be4b6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2019-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lewis2019mlqa,
  author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
  eid = {arXiv: 1910.07475},
  journal = {arXiv preprint arXiv:1910.07475},
  title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
  year = {2019},
}
""",
    )
