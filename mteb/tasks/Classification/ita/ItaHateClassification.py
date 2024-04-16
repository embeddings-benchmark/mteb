from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ItaHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ItaHateClassification",
        dataset={
            "path": "Paul/hatecheck-italian",
            "revision": "21e3d5c827cb60619a89988b24979850a7af85a5",
        },
        description="""Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate. Multilingual datase released as 10 unilingual models
                     """,
        reference="https://aclanthology.org/2022.woah-1.15/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        date=("2021-11-01", "2022-02-28"),
        form=["written"],
        domains=None,
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation="created",
        bibtex_citation="""
        @inproceedings{rottger-etal-2022-multilingual,
            title = "Multilingual {H}ate{C}heck: Functional Tests for Multilingual Hate Speech Detection Models",
            author = {R{\"o}ttger, Paul  and
            Seelawi, Haitham  and
            Nozza, Debora  and
            Talat, Zeerak  and
            Vidgen, Bertie},
            editor = "Narang, Kanika  and
            Mostafazadeh Davani, Aida  and
            Mathias, Lambert  and
            Vidgen, Bertie  and
            Talat, Zeerak",
            booktitle = "Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH)",
            month = jul,
            year = "2022",
            address = "Seattle, Washington (Hybrid)",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.woah-1.15",
            doi = "10.18653/v1/2022.woah-1.15",
            pages = "154--169",
            abstract = "Hate speech detection models are typically evaluated on held-out test sets. However, this risks painting an incomplete and potentially misleading picture of model performance because of increasingly well-documented systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, recent research has thus introduced functional tests for hate speech detection models. However, these tests currently only exist for English-language content, which means that they cannot support the development of more effective models in other languages spoken by billions across the world. To help address this issue, we introduce Multilingual HateCheck (MHC), a suite of functional tests for multilingual hate speech detection models. MHC covers 34 functionalities across ten languages, which is more languages than any other hate speech dataset. To illustrate MHC{'}s utility, we train and test a high-performing multilingual hate speech detection model, and reveal critical model weaknesses for monolingual and cross-lingual applications.",
        }
        """,
        n_samples={"test": 1845},
        avg_character_length={"test": 50.4},
    )

    def dataset_transform(self):
        keep_cols = ["test_case", "label_gold"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [
            col for col in self.dataset["test"].column_names if col not in keep_cols
        ]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.dataset["test"].train_test_split(
            test_size=0.5, seed=42, stratify_by_column="functionality"
        ) # balanced sampling across types of hate speech
        self.dataset = self.dataset.remove_columns(remove_cols)
