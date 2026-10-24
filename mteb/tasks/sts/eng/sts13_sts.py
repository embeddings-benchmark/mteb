from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS13STS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STS13",
        dataset={
            "path": "mteb/sts13-sts",
            "revision": "7e90230a92c190f1bf69ae9002b8cea547a64cca",
        },
        description="SemEval STS 2013 dataset.",
        reference="https://www.aclweb.org/anthology/S13-1004/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-12-31"),
        domains=["Web", "News", "Non-fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{agirre-etal-2013-sem,
  address = {Atlanta, Georgia, USA},
  author = {Agirre, Eneko  and
Cer, Daniel  and
Diab, Mona  and
Gonzalez-Agirre, Aitor  and
Guo, Weiwei},
  booktitle = {Second Joint Conference on Lexical and Computational Semantics (*{SEM}), Volume 1: Proceedings of the Main Conference and the Shared Task: Semantic Textual Similarity},
  editor = {Diab, Mona  and
Baldwin, Tim  and
Baroni, Marco},
  month = jun,
  pages = {32--43},
  publisher = {Association for Computational Linguistics},
  title = {*{SEM} 2013 shared task: Semantic Textual Similarity},
  url = {https://aclanthology.org/S13-1004/},
  year = {2013},
}
""",
    )
