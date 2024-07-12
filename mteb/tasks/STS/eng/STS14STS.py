from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


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
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-08-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{bandhakavi-etal-2014-generating,
    title = "Generating a Word-Emotion Lexicon from {\#}Emotional Tweets",
    author = "Bandhakavi, Anil  and
      Wiratunga, Nirmalie  and
      P, Deepak  and
      Massie, Stewart",
    editor = "Bos, Johan  and
      Frank, Anette  and
      Navigli, Roberto",
    booktitle = "Proceedings of the Third Joint Conference on Lexical and Computational Semantics (*{SEM} 2014)",
    month = aug,
    year = "2014",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics and Dublin City University",
    url = "https://aclanthology.org/S14-1002",
    doi = "10.3115/v1/S14-1002",
    pages = "12--21",
}""",
        descriptive_stats={
            "n_samples": {"test": 7500},
            "avg_character_length": {"test": 54.3},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
