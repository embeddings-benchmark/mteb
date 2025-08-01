from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NoRecClassification(AbsTaskClassification):
    superseded_by = "NoRecClassification.v2"
    metadata = TaskMetadata(
        name="NoRecClassification",
        description="A Norwegian dataset for sentiment classification on review",
        reference="https://aclanthology.org/L18-1661/",
        dataset={
            # using the mini version to keep results ~comparable to the ScandEval benchmark
            "path": "mteb/norec_classification",
            "revision": "5b740b7c42c73d586420812a35745fc37118862f",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
        date=("1998-01-01", "2018-01-01"),  # based on plot in paper
        domains=["Written", "Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{velldal-etal-2018-norec,
  address = {Miyazaki, Japan},
  author = {Velldal, Erik  and
{\\O}vrelid, Lilja  and
Bergem, Eivind Alexander  and
Stadsnes, Cathrine  and
Touileb, Samia  and
J{\\o}rgensen, Fredrik},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Hasida, Koiti  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\\'e}l{\\`e}ne  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios  and
Tokunaga, Takenobu},
  month = may,
  publisher = {European Language Resources Association (ELRA)},
  title = {{N}o{R}e{C}: The {N}orwegian Review Corpus},
  url = {https://aclanthology.org/L18-1661},
  year = {2018},
}
""",
        prompt="Classify Norwegian reviews by sentiment",
    )


class NoRecClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NoRecClassification.v2",
        description="""A Norwegian dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://aclanthology.org/L18-1661/",
        dataset={
            # using the mini version to keep results ~comparable to the ScandEval benchmark
            "path": "mteb/no_rec",
            "revision": "10aae1fb3fe2c19888bd4ea11695bbf19aa8bed3",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
        date=("1998-01-01", "2018-01-01"),  # based on plot in paper
        domains=["Written", "Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{velldal-etal-2018-norec,
  address = {Miyazaki, Japan},
  author = {Velldal, Erik  and
{\\O}vrelid, Lilja  and
Bergem, Eivind Alexander  and
Stadsnes, Cathrine  and
Touileb, Samia  and
J{\\o}rgensen, Fredrik},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Hasida, Koiti  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\\'e}l{\\`e}ne  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios  and
Tokunaga, Takenobu},
  month = may,
  publisher = {European Language Resources Association (ELRA)},
  title = {{N}o{R}e{C}: The {N}orwegian Review Corpus},
  url = {https://aclanthology.org/L18-1661},
  year = {2018},
}
""",
        prompt="Classify Norwegian reviews by sentiment",
        adapted_from=["NoRecClassification"],
    )
