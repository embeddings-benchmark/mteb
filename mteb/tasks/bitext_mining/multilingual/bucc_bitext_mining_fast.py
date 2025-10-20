from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "de-en": ["deu-Latn", "eng-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}


_SPLITS = ["test"]


class BUCCBitextMiningFast(AbsTaskBitextMining):
    fast_loading = True
    metadata = TaskMetadata(
        name="BUCC.v2",
        dataset={
            "path": "mteb/bucc-bitext-mining",
            "revision": "1739dc11ffe9b7bfccd7f3d585aeb4c544fc6677",
        },
        description="BUCC bitext mining dataset",
        reference="https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2017-01-01", "2018-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{zweigenbaum-etal-2017-overview,
  address = {Vancouver, Canada},
  author = {Zweigenbaum, Pierre  and
Sharoff, Serge  and
Rapp, Reinhard},
  booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
  doi = {10.18653/v1/W17-2512},
  editor = {Sharoff, Serge  and
Zweigenbaum, Pierre  and
Rapp, Reinhard},
  month = aug,
  pages = {60--67},
  publisher = {Association for Computational Linguistics},
  title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
  url = {https://aclanthology.org/W17-2512},
  year = {2017},
}
""",
        adapted_from=["BUCC"],
    )
