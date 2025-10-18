from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "ar": ["ara-Arab"],
    "fr": ["fra-Latn"],
    "ru": ["rus-Cyrl"],
}


class HUMESTS22(AbsTaskSTS):
    fast_loading = True
    metadata = TaskMetadata(
        name="HUMESTS22",
        dataset={
            "path": "mteb/mteb-human-sts22-sts",
            "revision": "ab40ed76c4283318b7b146aff31f8cdefd6ebae0",
        },
        description="Human evaluation subset of SemEval 2022 Task 8: Multilingual News Article Similarity",
        reference="https://competitions.codalab.org/competitions/33835",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2020-01-01", "2020-06-11"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen-etal-2022-semeval,
  address = {Seattle, United States},
  author = {Chen, Xi  and
Zeynali, Ali  and
Camargo, Chico  and
Fl{\"o}ck, Fabian  and
Gaffney, Devin  and
Grabowicz, Przemyslaw  and
Hale, Scott  and
Jurgens, David  and
Samory, Mattia},
  booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  doi = {10.18653/v1/2022.semeval-1.155},
  editor = {Emerson, Guy  and
Schluter, Natalie  and
Stanovsky, Gabriel  and
Kumar, Ritesh  and
Palmer, Alexis  and
Schneider, Nathan  and
Singh, Siddharth  and
Ratan, Shyam},
  month = jul,
  pages = {1094--1106},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
  url = {https://aclanthology.org/2022.semeval-1.155},
  year = {2022},
}
""",
        adapted_from=["STS22"],
    )
    min_score = 1
    max_score = 4
