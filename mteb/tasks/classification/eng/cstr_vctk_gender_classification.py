from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CSTRVCTKGenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSTRVCTKGender",
        description="Gender classification from CSTR-VCTK dataset. This is a stratified and downsampled version of the original dataset. The dataset was recorded with 2 different microphones, and this mini version uniformly samples data from the 2 microphone types.",
        reference="https://datashare.ed.ac.uk/handle/10283/3443",
        dataset={
            "path": "mteb/cstr-vctk-gender-mini",
            "revision": "8c7429cbb5c01d9327cff77dad5cbf813ecddc13",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-11-13", "2019-11-13"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Gender Classification"],
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
    label_column_name: str = "gender_id"
