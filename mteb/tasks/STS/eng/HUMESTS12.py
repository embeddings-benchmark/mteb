from __future__ import annotations

from mteb.abstasks import AbsTaskAnySTS
from mteb.abstasks.task_metadata import TaskMetadata


class HUMESTS12(AbsTaskAnySTS):
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
  abstract = {Semantic Textual Similarity (STS) measures the degree of semantic equivalence between two texts. This paper presents the results of the STS pilot task in Semeval. The training data contained 2000 sentence pairs from previously existing paraphrase datasets and machine translation evaluation resources. The test data also comprised 2000 sentences pairs for those datasets, plus two surprise datasets with 400 pairs from a different machine translation evaluation corpus and 750 pairs from a lexical resource mapping exercise. The similarity of pairs of sentences was rated on a 0-5 scale (low to high similarity) by human judges using Amazon Mechanical Turk, with high Pearson correlation scores, around 90\%. 35 teams participated in the task, submitting 88 runs. The best results scored a Pearson correlation >80\%, well above a simple lexical baseline that only scored a 31\% correlation. This pilot task opens an exciting way ahead, although there are still open issues, specially the evaluation metric.},
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
