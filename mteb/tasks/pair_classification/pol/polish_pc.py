from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SickePLPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICK-E-PL",
        dataset={
            "path": "PL-MTEB/sicke-pl-pairclassification",
            "revision": "71bba34b0ece6c56dfcf46d9758a27f7a90f17e9",
        },
        description="Polish version of SICK dataset for textual entailment.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Reviews"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{dadas-etal-2020-evaluation,
  address = {Marseille, France},
  author = {Dadas, Slawomir  and
Pere{\l}kiewicz, Micha{\l}  and
Po{\'s}wiata, Rafa{\l}},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {1674--1680},
  publisher = {European Language Resources Association},
  title = {Evaluation of Sentence Representations in {P}olish},
  url = {https://aclanthology.org/2020.lrec-1.207},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class PpcPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PpcPC",
        dataset={
            "path": "PL-MTEB/ppc-pairclassification",
            "revision": "2c7d2df57801a591f6b1e3aaf042e7a04ec7d9f2",
        },
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=None,
        domains=[
            "Fiction",
            "Non-fiction",
            "Web",
            "Written",
            "Spoken",
            # likely:
            "Social",
            "News",
        ],  # opensubtitles, CCmatrix
        task_subtypes=[],
        license="gpl-3.0",
        annotations_creators="derived",  # mined
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{dadas2022training,
  archiveprefix = {arXiv},
  author = {Sławomir Dadas},
  eprint = {2207.12759},
  primaryclass = {cs.CL},
  title = {Training Effective Neural Sentence Encoders from Automatically Mined Paraphrases},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class CdscePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CDSC-E",
        dataset={
            "path": "PL-MTEB/cdsce-pairclassification",
            "revision": "0a3d4aa409b22f80eb22cbf59b492637637b536d",
        },
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wroblewska-krasnowska-kieras-2017-polish,
  address = {Vancouver, Canada},
  author = {Wr{\'o}blewska, Alina  and
Krasnowska-Kiera{\'s}, Katarzyna},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/P17-1073},
  editor = {Barzilay, Regina  and
Kan, Min-Yen},
  month = jul,
  pages = {784--792},
  publisher = {Association for Computational Linguistics},
  title = {{P}olish evaluation dataset for compositional distributional semantics models},
  url = {https://aclanthology.org/P17-1073},
  year = {2017},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class PscPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PSC",
        dataset={
            "path": "PL-MTEB/psc-pairclassification",
            "revision": "d05a294af9e1d3ff2bfb6b714e08a24a6cabc669",
        },
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",  # and https://zil.ipipan.waw.pl/PolishSummariesCorpus
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=("1996-01-01", "2003-01-01"),  # from the paper
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ogrodniczuk-kopec-2014-polish,
  address = {Reykjavik, Iceland},
  author = {Ogrodniczuk, Maciej  and
Kope{\'c}, Mateusz},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Declerck, Thierry  and
Loftsson, Hrafn  and
Maegaard, Bente  and
Mariani, Joseph  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  month = may,
  pages = {3712--3715},
  publisher = {European Language Resources Association (ELRA)},
  title = {The {P}olish Summaries Corpus},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf},
  year = {2014},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
