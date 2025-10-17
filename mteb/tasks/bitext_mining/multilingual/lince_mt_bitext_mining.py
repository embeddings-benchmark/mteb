from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "eng-eng_hin": ["eng-Latn", "hin-Latn"],
}


class LinceMTBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="LinceMTBitextMining",
        dataset={
            "path": "gentaiscool/bitext_lincemt_miners",
            "revision": "483f2494f76fdc04acbbdbbac129de1925b34215",
        },
        description="LinceMT is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.",
        reference="https://ritual.uh.edu/lince/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2019-01-01", "2020-01-01"),
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{aguilar2020lince,
  author = {Aguilar, Gustavo and Kar, Sudipta and Solorio, Thamar},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages = {1803--1813},
  title = {LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation},
  year = {2020},
}
""",
    )
