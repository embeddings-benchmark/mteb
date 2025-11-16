from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PatentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PatentClassification",
        description="Classification Dataset of Patents and Abstract",
        dataset={
            "path": "mteb/PatentClassification",
            "revision": "6bd77eb030ab3bfbf1e6f7a2b069979daf167311",
        },
        reference="https://aclanthology.org/P19-1212.pdf",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-11-05", "2022-10-22"),
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sharma-etal-2019-bigpatent,
  address = {Florence, Italy},
  author = {Sharma, Eva  and
Li, Chen  and
Wang, Lu},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/P19-1212},
  editor = {Korhonen, Anna  and
Traum, David  and
M{\`a}rquez, Llu{\'\i}s},
  month = jul,
  pages = {2204--2213},
  publisher = {Association for Computational Linguistics},
  title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
  url = {https://aclanthology.org/P19-1212},
  year = {2019},
}
""",
        superseded_by="PatentClassification.v2",
    )


class PatentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PatentClassification.v2",
        description="Classification Dataset of Patents and Abstract This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/patent",
            "revision": "f5e5c81286448c68264300fe1e6f3de599922890",
        },
        reference="https://aclanthology.org/P19-1212.pdf",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-11-05", "2022-10-22"),
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sharma-etal-2019-bigpatent,
  address = {Florence, Italy},
  author = {Sharma, Eva  and
Li, Chen  and
Wang, Lu},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/P19-1212},
  editor = {Korhonen, Anna  and
Traum, David  and
M{\`a}rquez, Llu{\'\i}s},
  month = jul,
  pages = {2204--2213},
  publisher = {Association for Computational Linguistics},
  title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
  url = {https://aclanthology.org/P19-1212},
  year = {2019},
}
""",
    )
