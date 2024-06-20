from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

_LANGUAGES = {
    "eng-eng_hin": ["eng-Latn", "hin-Latn"],
}


class LinceMTBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="LinceMTBitextMining",
        dataset={
            "path": "gentaiscool/bitext_lincemt_miners",
            "revision": "483f2494f76fdc04acbbdbbac129de1925b34215",
        },
        description="LinceMT is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.",
        reference="https://ritual.uh.edu/lince/",
        type="BitextMining",
        category="s2s",
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2019-01-01", "2020-01-01"),
        form=["written"],
        domains=["Social"],
        task_subtypes=[],
        license="Unknown",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{aguilar2020lince,
        title={LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation},
        author={Aguilar, Gustavo and Kar, Sudipta and Solorio, Thamar},
        booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
        pages={1803--1813},
        year={2020}
        }
        """,
        n_samples={"train": 8060},
        avg_character_length={"train": 58.67},
    )
