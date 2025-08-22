from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArmenianParaphrasePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ArmenianParaphrasePC",
        description="asparius/Armenian-Paraphrase-PC",
        reference="https://github.com/ivannikov-lab/arpa-paraphrase-corpus",
        dataset={
            "path": "asparius/Armenian-Paraphrase-PC",
            "revision": "f43b4f32987048043a8b31e5e26be4d360c2438f",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hye-Armn"],
        main_score="max_ap",
        date=("2021-01-01", "2022-04-06"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{malajyan2020arpa,
  archiveprefix = {arXiv},
  author = {Arthur Malajyan and Karen Avetisyan and Tsolak Ghukasyan},
  eprint = {2009.12615},
  primaryclass = {cs.CL},
  title = {ARPA: Armenian Paraphrase Detection Corpus and Models},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
