from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "ru": ["rus-Cyrl"],
    "sv": ["swe-Latn"],
}


class OpusparcusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="OpusparcusPC",
        dataset={
            "path": "mteb/OpusparcusPC",
            "revision": "e95831b0fa902edc955f3b60e3e63b349bb7bd02",
        },
        description="Opusparcus is a paraphrase corpus for six European language: German, English, Finnish, French, Russian, and Swedish. The paraphrases consist of subtitles from movies and TV shows.",
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test.full", "validation.full"],
        eval_langs=_LANGUAGES,
        main_score="max_ap",
        date=("2013-01-01", "2015-12-31"),
        domains=["Spoken", "Spoken"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{creutz2018open,
  author = {Creutz, Mathias},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  collection = {LREC},
  doi = {10.63317/4kki5rwb9a2a},
  issn = {2522-2686},
  month = {May},
  publisher = {European Language Resources Association (ELRA)},
  series = {LREC},
  title = {Open Subtitles Paraphrase Corpus for Six Languages},
  url = {https://doi.org/10.63317/4kki5rwb9a2a},
  year = {2018},
}
""",
    )
