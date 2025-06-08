from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS12STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS12",
        dataset={
            "path": "mteb/sts12-sts",
            "revision": "a0d554a64d88156834ff5ae9920b964011b16384",
        },
        description="SemEval-2012 Task 6.",
        reference="https://www.aclweb.org/anthology/S12-1051.pdf",
        type="STS",
        category="s2s",
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
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
