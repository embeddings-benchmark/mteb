from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenkSlClassification(AbsTaskClassification):
    superseded_by = "FrenkSlClassification.v2"
    metadata = TaskMetadata(
        name="FrenkSlClassification",
        description="Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.",
        dataset={
            "path": "classla/FRENK-hate-sl",
            "revision": "37c8b42c63d4eb75f549679158a85eb5bd984caa",
            "trust_remote_code": True,
        },
        reference="https://arxiv.org/pdf/1906.02045",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slv-Latn"],
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

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class FrenkSlClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkSlClassification.v2",
        description="""Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        dataset={
            "path": "mteb/frenk_sl",
            "revision": "3b69facc14651fbd152fda173683a7ecf9125b82",
        },
        reference="https://arxiv.org/pdf/1906.02045",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slv-Latn"],
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
        adapted_from=["FrenkSlClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
