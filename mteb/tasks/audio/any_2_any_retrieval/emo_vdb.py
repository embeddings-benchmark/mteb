from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EmoVDBA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EmoVDBA2TRetrieval",
        description="Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.",
        reference="https://github.com/numediart/EmoV-DB?tab=readme-ov-file",
        dataset={
            "path": "mteb/EmoV_DB_a2t",
            "revision": "5a25db2a3f435e28b36576d7cf68e352da901251",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotional Speech Retrieval"],
        license="https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{adigwe2018emotional,
  author = {Adigwe, Adaeze and Tits, No{\'e} and Haddad, Kevin El and Ostadabbas, Sarah and Dutoit, Thierry},
  journal = {arXiv preprint arXiv:1806.09514},
  title = {The emotional voices database: Towards controlling the emotion dimension in voice generation systems},
  year = {2018},
}
""",
    )


class EmoVDBT2ARetrieval(AbsTaskRetrieval):
    """Text-to-audio retrieval on the EmoV-DB emotional voice captions ↔ audio pairs."""

    metadata = TaskMetadata(
        name="EmoVDBT2ARetrieval",
        description="Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.",
        reference="https://github.com/numediart/EmoV-DB?tab=readme-ov-file",
        dataset={
            "path": "mteb/EmoV_DB_t2a",
            "revision": "692b1871d0b7d02ea4717b23882b62b7e27c23cd",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotional Speech Retrieval"],
        license="https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{adigwe2018emotional,
  author = {Adigwe, Adaeze and Tits, No{\'e} and Haddad, Kevin El and Ostadabbas, Sarah and Dutoit, Thierry},
  journal = {arXiv preprint arXiv:1806.09514},
  title = {The emotional voices database: Towards controlling the emotion dimension in voice generation systems},
  year = {2018},
}
""",
    )
