from __future__ import annotations

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "Danish": ["dan-Latn"],
    "Norwegian_b": ["nob-Latn"],
    "Norwegian_n": ["nno-Latn"],
    "Swedish": ["swe-Latn"],
}


class ScalaClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="MultiScalaClassification",
        description="""ScaLa a linguistic acceptability dataset for the mainland Scandinavian languages automatically constructed from dependency annotations in Universal Dependencies Treebanks. 
        Published as part of 'ScandEval: A Benchmark for Scandinavian Natural Language Processing'""",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "mteb/multilingual-scala-classification",
            "revision": "ec85bb6c69679ed15ac66c0bf6e180bf563eb137",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="accuracy",
        date=("1990-01-01", "2023-01-01") # derived from dependency treebank, this a the best guess
        form=["written"],
        domains=None,
        task_subtypes=["Linguistic acceptability"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=None,
        text_creation="created",
        bibtex_citation="""
        @inproceedings{nielsen-2023-scandeval,
            title = "{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing",
            author = "Nielsen, Dan",
            editor = {Alum{\"a}e, Tanel  and
            Fishel, Mark},
            booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
            month = may,
            year = "2023",
            address = "T{\'o}rshavn, Faroe Islands",
            publisher = "University of Tartu Library",
            url = "https://aclanthology.org/2023.nodalida-1.20",
            pages = "185--201",
        }""",
        n_samples={"test": 1024},
        avg_character_length={"test": 102.72},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        for lang in self.dataset.keys():
            # convert label to a 0/1 label
            labels = self.dataset[lang]["train"]["label"]  # type: ignore
            lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
            self.dataset[lang] = self.dataset[lang].map(
                lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
            )
