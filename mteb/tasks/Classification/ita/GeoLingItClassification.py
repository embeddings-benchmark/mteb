from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

class GeoLingItClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GeoLingItClassification",
        dataset={
            "path": "MattiaSangermano/GeoLingIt",
            "revision": "ac4c2cec8497bff51bd5f19fba03943313b57c50",
        },
        description="GeoLingIt is a dataset for studying the geolocation of linguistic variation in Italy using social media posts that exhibit non-standard Italian language",
        reference="https://github.com/dhfbk/geolingit-evalita2023",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation","test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        domains=["Written","Social"],
        task_subtypes=["Language identification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ramponi-casula-2023-diatopit,
            title = "{D}iatop{I}t: A Corpus of Social Media Posts for the Study of Diatopic Language Variation in {I}taly",
            author = "Ramponi, Alan  and
            Casula, Camilla",
            booktitle = "Tenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2023)",
            month = may,
            year = "2023",
            address = "Dubrovnik, Croatia",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.vardial-1.19",
            pages = "187--199",
            }""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("region", "label")
        unused_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(unused_cols)