from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = {
    "spanish": ["spa-Latn"],
    "catalan": ["cat-Latn"],
}


class CataloniaTweetClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CataloniaTweetClassification",
        description="This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter messages for automatic stance detection. The data was collected over 12 days during February and March of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia. Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance towards the target - independence of Catalonia.",
        reference="https://aclanthology.org/2020.lrec-1.171/",
        dataset={
            "path": "mteb/CataloniaTweetClassification",
            "revision": "a160e486a21cea511a4b23a90e6bd6f7920d3a8f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=_LANGS,
        main_score="accuracy",
        date=("2018-09-01", "2029-03-30"),
        domains=["Social", "Government", "Written"],
        task_subtypes=["Political classification"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{zotova-etal-2020-multilingual,
  author = {Zotova, Elena  and
Agerri, Rodrigo  and
Nu{\~n}ez, Manuel  and
Rigau, German},
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
  month = may,
  pages = {1368--1375},
  publisher = {European Language Resources Association},
  title = {Multilingual Stance Detection in Tweets: The {C}atalonia Independence Corpus},
  year = {2020},
}
""",
    )
