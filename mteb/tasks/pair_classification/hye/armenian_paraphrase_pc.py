from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ArmenianParaphrasePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ArmenianParaphrasePC",
        description="asparius/Armenian-Paraphrase-PC",
        reference="https://github.com/ivannikov-lab/arpa-paraphrase-corpus",
        dataset={
            "path": "mteb/ArmenianParaphrasePC",
            "revision": "594f25dfef459cbfa30fc1765a6fb2053244f85b",
        },
        type="PairClassification",
        category="t2t",
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
