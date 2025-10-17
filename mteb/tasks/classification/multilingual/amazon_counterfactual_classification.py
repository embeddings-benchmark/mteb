from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonCounterfactualClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonCounterfactualClassification",
        dataset={
            "path": "mteb/amazon_counterfactual",
            "revision": "1f7e6a9d6fa6e64c53d146e428565640410c0df1",
        },
        description=(
            "A collection of Amazon customer reviews annotated for counterfactual detection pair classification."
        ),
        reference="https://arxiv.org/abs/2104.06893",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={
            "en-ext": ["eng-Latn"],
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "ja": ["jpn-Jpan"],
        },
        main_score="accuracy",
        date=(
            "2018-01-01",
            "2021-12-31",
        ),  # Estimated range for the collection of Amazon reviews
        domains=["Reviews", "Written"],
        task_subtypes=["Counterfactual Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{oneill-etal-2021-wish,
  address = {Online and Punta Cana, Dominican Republic},
  author = {O{'}Neill, James  and
Rozenshtein, Polina  and
Kiryo, Ryuichi  and
Kubota, Motoko  and
Bollegala, Danushka},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2021.emnlp-main.568},
  editor = {Moens, Marie-Francine  and
Huang, Xuanjing  and
Specia, Lucia  and
Yih, Scott Wen-tau},
  month = nov,
  pages = {7092--7108},
  publisher = {Association for Computational Linguistics},
  title = {{I} Wish {I} Would Have Loved This One, But {I} Didn{'}t {--} A Multilingual Dataset for Counterfactual Detection in Product Review},
  url = {https://aclanthology.org/2021.emnlp-main.568},
  year = {2021},
}
""",
        prompt="Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    )

    samples_per_label = 32
