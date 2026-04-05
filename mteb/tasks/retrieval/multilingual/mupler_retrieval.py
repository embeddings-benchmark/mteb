from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "pt": ["por-Latn"],
    "pl": ["pol-Latn"],
    "fr": ["fra-Latn"],
    "sl": ["slv-Latn"],
    "sv": ["swe-Latn"],
    "sk": ["slk-Latn"],
    "it": ["ita-Latn"],
    "el": ["ell-Latn"],
    "lt": ["lit-Latn"],
    "lv": ["lav-Latn"],
    "es": ["spa-Latn"],
    "nl": ["nld-Latn"],
}


class MuPLeRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MuPLeR-retrieval",
        description="MuPLeR-retrieval is a multilingual, parallel legal dataset designed for evaluating retrieval and cross-lingual retrieval tasks. Dataset contains 10,000 human-translated parallel passages (derived from the European Union's DGT-Acquis corpus) & 200 parallel queries (synthetic) across 14 European languages.",
        reference="https://link.springer.com/article/10.1007/s10579-014-9277-0",
        dataset={
            "path": "mteb/MuPLeR-retrieval",
            "revision": "43dff1bca775c1298511f6776f0ac1d5cc144ccb",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2026-12-01", "2026-03-26"),
        domains=["Legal"],
        task_subtypes=[],
        license="eupl-1.2",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        is_public=True,
        bibtex_citation=r"""
@article{steinberger2014overview,
  author = {Steinberger, Ralf and Ebrahim, Mohamed and Poulis, Alexandros and {Carrasco-Benitez}, Manuel and Schl{\"u}ter, Patrick and Przybyszewski, Marek and Gilbro, Signe},
  doi = {10.1007/s10579-014-9277-0},
  issn = {1574-0218},
  journal = {Language Resources and Evaluation},
  keywords = {DCEP,DGT-Acquis,DGT-TM,EAC-TM,ECDC-TM,Eur-Lex,European Union,EuroVoc,Highly multilingual,JRC EuroVoc Indexer JEX,JRC-Acquis,Linguistic resources,Parallel corpora,Translation memory},
  langid = {english},
  language = {en},
  month = dec,
  number = {4},
  pages = {679--707},
  title = {An Overview of the {{European Union}}'s Highly Multilingual Parallel Corpora},
  urldate = {2026-03-29},
  volume = {48},
  year = {2014},
}
""",
    )
