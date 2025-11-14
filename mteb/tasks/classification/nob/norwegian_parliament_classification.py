from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NorwegianParliamentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NorwegianParliamentClassification",
        description="Norwegian parliament speeches annotated for sentiment",
        reference="https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
        dataset={
            "path": "mteb/NorwegianParliamentClassification",
            "revision": "3047317d2586abb183293f92b1b7d66d1c9ec81a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["nob-Latn"],
        # assumed to be bokmål
        main_score="accuracy",
        date=("1999-01-01", "2016-01-01"),  # based on dates within the dataset
        domains=["Government", "Spoken"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="derived",  # based on the speaker affiliation
        dialect=[],  # unknown
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kummervold-etal-2021-operationalizing,
  address = {Reykjavik, Iceland (Online)},
  author = {Kummervold, Per E  and
De la Rosa, Javier  and
Wetjen, Freddy  and
Brygfjeld, Svein Arne},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {20--29},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
  url = {https://aclanthology.org/2021.nodalida-main.3},
  year = {2021},
}
""",
        prompt="Classify parliament speeches in Norwegian based on political affiliation",
        superseded_by="NorwegianParliamentClassification.v2",
    )


class NorwegianParliamentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NorwegianParliamentClassification.v2",
        description="Norwegian parliament speeches annotated for sentiment This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
        dataset={
            "path": "mteb/norwegian_parliament",
            "revision": "7f2012a878e67486ac871cb450d6ef0dc2ebed7f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["nob-Latn"],
        # assumed to be bokmål
        main_score="accuracy",
        date=("1999-01-01", "2016-01-01"),  # based on dates within the dataset
        domains=["Government", "Spoken"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="derived",  # based on the speaker affiliation
        dialect=[],  # unknown
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kummervold-etal-2021-operationalizing,
  address = {Reykjavik, Iceland (Online)},
  author = {Kummervold, Per E  and
De la Rosa, Javier  and
Wetjen, Freddy  and
Brygfjeld, Svein Arne},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {20--29},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
  url = {https://aclanthology.org/2021.nodalida-main.3},
  year = {2021},
}
""",
        prompt="Classify parliament speeches in Norwegian based on political affiliation",
        adapted_from=["NorwegianParliamentClassification"],
    )
