# SuperLIM tasks


from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DalajClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DalajClassification",
        dataset={
            "path": "mteb/DalajClassification",
            "revision": "2a460943e43451e4c9aed3c0adafc83ccd6e3ee1",
        },
        description="A Swedish dataset for linguistic acceptability. Available as a part of Superlim.",
        reference="https://spraakbanken.gu.se/en/resources/superlim",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2017-01-01", "2020-12-31"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Linguistic acceptability"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{2105.06681,
  address = {Online},
  archiveprefix = {arXiv},
  author = {Volodina, Elena and Mohammed, Yousuf Ali and Klezl, Julia},
  booktitle = {Proceedings of the 10th Workshop on {NLP} for Computer Assisted Language Learning},
  editor = {Alfter, David and Volodina, Elena and Pilan, Ildik{\'o} and Gra{\"e}n, Johannes and Borin, Lars},
  eprint = {2105.06681},
  month = {may},
  pages = {28--37},
  publisher = {LiU Electronic Press},
  title = {{D}a{LAJ} {--} a dataset for linguistic acceptability judgments for {S}wedish},
  url = {https://aclanthology.org/2021.nlp4call-1.3/},
  year = {2021},
}
""",
        prompt="Classify texts based on linguistic acceptability in Swedish",
        superseded_by="DalajClassification.v2",
    )

    samples_per_label = 16


class DalajClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DalajClassification.v2",
        dataset={
            "path": "mteb/dalaj",
            "revision": "ecf6f2d83e8e85816ec3974896557a4aafce4f3e",
            "name": "dalaj",
        },
        description="A Swedish dataset for linguistic acceptability. Available as a part of Superlim. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://spraakbanken.gu.se/en/resources/superlim",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2017-01-01", "2020-12-31"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Linguistic acceptability"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{2105.06681,
  address = {Online},
  archiveprefix = {arXiv},
  author = {Volodina, Elena and Mohammed, Yousuf Ali and Klezl, Julia},
  booktitle = {Proceedings of the 10th Workshop on {NLP} for Computer Assisted Language Learning},
  editor = {Alfter, David and Volodina, Elena and Pilan, Ildik{\'o} and Gra{\"e}n, Johannes and Borin, Lars},
  eprint = {2105.06681},
  month = {may},
  pages = {28--37},
  publisher = {LiU Electronic Press},
  title = {{D}a{LAJ} {--} a dataset for linguistic acceptability judgments for {S}wedish},
  url = {https://aclanthology.org/2021.nlp4call-1.3/},
  year = {2021},
}
""",
        prompt="Classify texts based on linguistic acceptability in Swedish",
        adapted_from=["DalajClassification"],
    )

    samples_per_label = 16
