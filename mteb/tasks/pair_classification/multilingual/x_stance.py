from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class XStance(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XStance",
        dataset={
            "path": "mteb/XStance",
            "revision": "33c45c244e7c974f0c206372285a37e3f000f65a",
        },
        description="A Multilingual Multi-Target Dataset for Stance Detection in French, German, and Italian.",
        reference="https://github.com/ZurichNLP/xstance",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
        },
        main_score="max_ap",
        date=("2011-01-01", "2020-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Political classification"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{vamvas2020xstance,
  address = {Zurich, Switzerland},
  author = {Vamvas, Jannis and Sennrich, Rico},
  booktitle = {Proceedings of the 5th Swiss Text Analytics Conference (SwissText)  16th Conference on Natural Language Processing (KONVENS)},
  month = {jun},
  title = {{X-Stance}: A Multilingual Multi-Target Dataset for Stance Detection},
  url = {http://ceur-ws.org/Vol-2624/paper9.pdf},
  year = {2020},
}
""",
    )
