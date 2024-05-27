from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ImdbClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ImdbClassification",
        description="Large Movie Review Dataset",
        dataset={
            "path": "mteb/imdb",
            "revision": "3d86128a09e091d6018b6d26cad27f2739fc2db7",
        },
        reference="http://www.aclweb.org/anthology/P11-1015",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2000-01-01",
            "2010-12-31",
        ),  # Estimated range for the collection of movie reviews
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{maas-etal-2011-learning,
    title = "Learning Word Vectors for Sentiment Analysis",
    author = "Maas, Andrew L.  and
      Daly, Raymond E.  and
      Pham, Peter T.  and
      Huang, Dan  and
      Ng, Andrew Y.  and
      Potts, Christopher",
    editor = "Lin, Dekang  and
      Matsumoto, Yuji  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    address = "Portland, Oregon, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P11-1015",
    pages = "142--150",
}""",
        n_samples={"test": 25000},
        avg_character_length={"test": 1293.8},
    )
