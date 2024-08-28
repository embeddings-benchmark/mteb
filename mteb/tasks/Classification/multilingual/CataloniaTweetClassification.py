from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "spanish": ["spa-Latn"],
    "catalan": ["cat-Latn"],
}


class CataloniaTweetClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="CataloniaTweetClassification",
        description="""This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter
        messages for automatic stance detection. The data was collected over 12 days during February and March
        of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia.
        Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance
        towards the target - independence of Catalonia.
        """,
        reference="https://aclanthology.org/2020.lrec-1.171/",
        dataset={
            "path": "community-datasets/catalonia_independence",
            "revision": "cf24d44e517efa534f048e5fc5981f399ed25bee",
        },
        type="Classification",
        category="s2s",
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
        bibtex_citation="""@inproceedings{zotova-etal-2020-multilingual,
            title = "Multilingual Stance Detection in Tweets: The {C}atalonia Independence Corpus",
            author = "Zotova, Elena  and
            Agerri, Rodrigo  and
            Nu{\~n}ez, Manuel  and
            Rigau, German",
            editor = "Calzolari, Nicoletta  and
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
            Piperidis, Stelios",
            booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
            month = may,
            year = "2020",
            publisher = "European Language Resources Association",
            pages = "1368--1375",
            ISBN = "979-10-95546-34-4",
        }""",
        descriptive_stats={
            "n_samples": {"validation": 2000, "test": 2000},
            "avg_character_length": {"validation": 202.61, "test": 200.49},
        },
    )

    def dataset_transform(self):
        for lang in self.dataset.keys():
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"TWEET": "text", "LABEL": "label"}
            )
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang],
                seed=self.seed,
                splits=["validation", "test"],
                n_samples=2000,
            )
            self.dataset[lang] = self.dataset[lang].remove_columns(["id_str"])
