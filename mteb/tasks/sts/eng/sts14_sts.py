from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS14STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS14",
        dataset={
            "path": "mteb/sts14-sts",
            "revision": "6031580fec1f6af667f0bd2da0a551cf4f0b2375",
        },
        description="SemEval STS 2014 dataset. Currently only the English dataset",
        reference="https://www.aclweb.org/anthology/S14-1002",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-08-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bandhakavi-etal-2014-generating,
  address = {Dublin, Ireland},
  author = {Bandhakavi, Anil  and
Wiratunga, Nirmalie  and
P, Deepak  and
Massie, Stewart},
  booktitle = {Proceedings of the Third Joint Conference on Lexical and Computational Semantics (*{SEM} 2014)},
  doi = {10.3115/v1/S14-1002},
  editor = {Bos, Johan  and
Frank, Anette  and
Navigli, Roberto},
  month = aug,
  pages = {12--21},
  publisher = {Association for Computational Linguistics and Dublin City University},
  title = {Generating a Word-Emotion Lexicon from {\#}Emotional Tweets},
  url = {https://aclanthology.org/S14-1002},
  year = {2014},
}
""",
    )

    min_score = 0
    max_score = 5
