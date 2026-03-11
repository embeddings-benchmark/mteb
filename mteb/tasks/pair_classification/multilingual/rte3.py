from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
}


class RTE3(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RTE3",
        dataset={
            "path": "mteb/RTE3",
            "revision": "54ea0052267265f4906dd77b0a3d041d301a5ee6",
        },
        description="Recognising Textual Entailment Challenge (RTE-3) aim to provide the NLP community with a benchmark to test progress in recognizing textual entailment",
        reference="https://aclanthology.org/W07-1401/",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="max_ap",
        date=("2023-03-25", "2024-04-15"),
        domains=["News", "Web", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{giampiccolo-etal-2007-third,
  address = {Prague},
  author = {Giampiccolo, Danilo  and
Magnini, Bernardo  and
Dagan, Ido  and
Dolan, Bill},
  booktitle = {Proceedings of the {ACL}-{PASCAL} Workshop on Textual Entailment and Paraphrasing},
  month = jun,
  pages = {1--9},
  publisher = {Association for Computational Linguistics},
  title = {The Third {PASCAL} Recognizing Textual Entailment Challenge},
  url = {https://aclanthology.org/W07-1401},
  year = {2007},
}
""",
        # sum of 4 languages after neutral filtering
    )
