from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenkEnClassification(AbsTaskClassification):
    superseded_by = "FrenkEnClassification.v2"
    metadata = TaskMetadata(
        name="FrenkEnClassification",
        description="English subset of the FRENK dataset",
        dataset={
            "path": "classla/FRENK-hate-en",
            "revision": "52483dba0ff23291271ee9249839865e3c3e7e50",
            "trust_remote_code": True,
        },
        reference="https://arxiv.org/abs/1906.02045",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-05-28", "2021-05-28"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{ljubešić2019frenk,
  archiveprefix = {arXiv},
  author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
  eprint = {1906.02045},
  primaryclass = {cs.CL},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://arxiv.org/abs/1906.02045},
  year = {2019},
}
""",
    )


class FrenkEnClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkEnClassification.v2",
        description="""English subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        dataset={
            "path": "mteb/frenk_en",
            "revision": "630d941b6e0879a7238da89af6bfe1b1eb27ca0f",
        },
        reference="https://arxiv.org/abs/1906.02045",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-05-28", "2021-05-28"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{ljubešić2019frenk,
  archiveprefix = {arXiv},
  author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
  eprint = {1906.02045},
  primaryclass = {cs.CL},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://arxiv.org/abs/1906.02045},
  year = {2019},
}
""",
        adapted_from=["FrenkEnClassification"],
    )
