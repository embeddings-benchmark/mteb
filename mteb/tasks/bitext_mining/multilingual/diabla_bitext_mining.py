from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class DiaBLaBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="DiaBlaBitextMining",
        dataset={
            "path": "mteb/DiaBlaBitextMining",
            "revision": "c458e9bf4306d6380604462926a38c34861b4d3b",
        },
        description="English-French Parallel Corpus. DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue.",
        reference="https://inria.hal.science/hal-03021633",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "fr-en": ["fra-Latn", "eng-Latn"],
            "en-fr": ["eng-Latn", "fra-Latn"],
        },
        main_score="f1",
        date=("2016-01-01", "2017-12-31"),
        domains=["Social", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{bawden2021diabla,
  author = {Bawden, Rachel and Bilinski, Eric and Lavergne, Thomas and Rosset, Sophie},
  journal = {Language Resources and Evaluation},
  number = {3},
  pages = {635--660},
  publisher = {Springer},
  title = {DiaBLa: a corpus of bilingual spontaneous written dialogues for machine translation},
  volume = {55},
  year = {2021},
}
""",
    )
