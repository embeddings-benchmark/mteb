from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class CovidDisinformationNLMultiLabelClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="CovidDisinformationNLMultiLabelClassification",
        dataset={
            "path": "clips/mteb-nl-COVID-19-disinformation",
            "revision": "7ad922bdef875db1f530847c6ffff05fc154f2e8",
        },
        description="The dataset is curated to address questions of interest to journalists, fact-checkers, "
        "social media platforms, policymakers, and the general public.",
        reference="https://aclanthology.org/2021.findings-emnlp.56.pdf",
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2020-01-01", "2021-04-01"),
        domains=["Web", "Social", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{alam-etal-2021-fighting-covid,
  address = {Punta Cana, Dominican Republic},
  author = {Alam, Firoj  and
Shaar, Shaden  and
Dalvi, Fahim  and
Sajjad, Hassan  and
Nikolov, Alex  and
Mubarak, Hamdy  and
Da San Martino, Giovanni  and
Abdelali, Ahmed  and
Durrani, Nadir  and
Darwish, Kareem  and
Al-Homaid, Abdulaziz  and
Zaghouani, Wajdi  and
Caselli, Tommaso  and
Danoe, Gijs  and
Stolk, Friso  and
Bruntink, Britt  and
Nakov, Preslav},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
  doi = {10.18653/v1/2021.findings-emnlp.56},
  editor = {Moens, Marie-Francine  and
Huang, Xuanjing  and
Specia, Lucia  and
Yih, Scott Wen-tau},
  month = nov,
  pages = {611--649},
  publisher = {Association for Computational Linguistics},
  title = {Fighting the {COVID}-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society},
  url = {https://aclanthology.org/2021.findings-emnlp.56/},
  year = {2021},
}
""",
        prompt={
            "query": "Classificeer COVID-19-gerelateerde sociale media-berichten in alle toepasselijke desinformatiecategorieën"
        },
    )

    def dataset_transform(self) -> None:
        labels = [
            "q2_label",
            "q3_label",
            "q4_label",
            "q5_label",
            "q6_label",
            "q7_label",
        ]
        _dataset = {}

        def map_labels(example):
            ml_labels = []
            for i, label in enumerate(labels):
                if example[label] == "yes":
                    ml_labels.append(i)
            return {"label": ml_labels}

        for split in self.dataset:
            self.dataset[split] = self.dataset[split].filter(
                lambda ex: ex["q1_label"] == "yes"
            )
            self.dataset[split] = self.dataset[split].map(map_labels)
