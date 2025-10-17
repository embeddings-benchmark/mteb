from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class HUMESTS12(AbsTaskSTS):
    metadata = TaskMetadata(
        name="HUMESTS12",
        dataset={
            "path": "mteb/mteb-human-sts12-sts",
            "revision": "76cbf76792ec03cb1f76dc6ada05abcb23c82c0c",
        },
        description="Human evaluation subset of SemEval-2012 Task 6.",
        reference="https://www.aclweb.org/anthology/S12-1051.pdf",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2005-01-01", "2012-12-31"),
        domains=["Encyclopaedic", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{10.5555/2387636.2387697,
  address = {USA},
  author = {Agirre, Eneko and Diab, Mona and Cer, Daniel and Gonzalez-Agirre, Aitor},
  booktitle = {Proceedings of the First Joint Conference on Lexical and Computational Semantics - Volume 1: Proceedings of the Main Conference and the Shared Task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation},
  location = {Montr\'{e}al, Canada},
  numpages = {9},
  pages = {385–393},
  publisher = {Association for Computational Linguistics},
  series = {SemEval '12},
  title = {SemEval-2012 task 6: a pilot on semantic textual similarity},
  year = {2012},
}
""",
        adapted_from=["STS12"],
    )
    min_score = 0
    max_score = 5
