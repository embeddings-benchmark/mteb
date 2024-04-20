from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PolHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolHateClassification",
        dataset={
            "path": "Paul/hatecheck-polish",
            "revision": "28d7098e2e5a211c4810d0a4d8deccc5889e55b6",
        },
        description="""Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate. Multilingual datase released as 10 unilingual models
                     """,
        reference="https://aclanthology.org/2022.woah-1.15/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2022-07-05", "2022-07-05"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
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
        n_samples={"test": 1907},
        avg_character_length={"test": 49.7},
    )

    def dataset_transform(self):
        keep_cols = ["test_case", "label_gold"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [
            col for col in self.dataset["test"].column_names if col not in keep_cols
        ]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.dataset.class_encode_column("functionality")
        self.dataset = self.dataset["test"].train_test_split(
            test_size=0.5, seed=42, stratify_by_column="functionality"
        )  # balanced sampling across types of hate speech
        self.dataset = self.dataset.remove_columns(remove_cols)


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from mteb import MTEB
    # Define the sentence-transformers model name
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=[PolHateClassification()])
    evaluation.run(model)
