from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

STS12_CITATION = r"""
@inproceedings{agirre-etal-2012-semeval,
  title = {{S}em{E}val-2012 Task 6: A Pilot on Semantic Textual Similarity},
  author = {Agirre, Eneko  and
      Cer, Daniel  and
      Diab, Mona  and
      Gonzalez-Agirre, Aitor},
  editor = {Agirre, Eneko  and
      Bos, Johan  and
      Diab, Mona  and
      Manandhar, Suresh  and
      Marton, Yuval  and
      Yuret, Deniz},
  booktitle = {*{SEM} 2012: The First Joint Conference on Lexical and Computational Semantics {--} Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth {I}nternational {W}orkshop on {S}emantic {E}valuation ({S}em{E}val 2012)},
  month = "7-8 " # jun,
  year = {2012},
  address = {Montr{\'e}al, Canada},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/S12-1051/},
  pages = {385--393},
}
"""


class STS12STS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STS12",
        dataset={
            "path": "mteb/sts12-sts",
            "revision": "a0d554a64d88156834ff5ae9920b964011b16384",
        },
        description="SemEval-2012 Task 6.",
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
        bibtex_citation=STS12_CITATION,
    )
