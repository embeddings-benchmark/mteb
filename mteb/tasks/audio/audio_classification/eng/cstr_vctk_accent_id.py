from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CSTRVCTKAccentID(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSTRVCTKAccentID",
        description="Gender classification from CSTR-VCTK dataset. This is a stratified and downsampled version of the original dataset. The dataset was recorded with 2 different microphones, and this mini version uniformly samples data from the 2 microphone types.",
        reference="https://datashare.ed.ac.uk/handle/10283/3443",
        dataset={
            "path": "mteb/cstr-vctk-accent-mini",
            "revision": "ceb854ae5018298bd1b5edb5983b78515ef8ded6",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2026-01-15", "2026-01-15"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Accent identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Yamagishi2019CSTRVC,
  author = {Junichi Yamagishi and Christophe Veaux and Kirsten MacDonald},
  title = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)},
  url = {https://api.semanticscholar.org/CorpusID:213060286},
  year = {2019},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "accents"
