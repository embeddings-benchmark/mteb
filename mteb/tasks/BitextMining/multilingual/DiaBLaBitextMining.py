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
@inproceedings{gonzalez2019diabla,
  author = {González, Matilde and García, Clara and Sánchez, Lucía},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  pages = {4192--4198},
  title = {DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation},
  year = {2019},
}
""",
    )
