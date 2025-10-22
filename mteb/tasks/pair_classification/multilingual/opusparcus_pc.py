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
@misc{creutz2018open,
  archiveprefix = {arXiv},
  author = {Mathias Creutz},
  eprint = {1809.06142},
  primaryclass = {cs.CL},
  title = {Open Subtitles Paraphrase Corpus for Six Languages},
  year = {2018},
}
""",
    )
