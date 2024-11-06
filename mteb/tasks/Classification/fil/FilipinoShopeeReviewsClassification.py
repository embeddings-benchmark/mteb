from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FilipinoShopeeReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FilipinoShopeeReviewsClassification",
        description="The Shopee reviews tl 15 dataset is constructed by randomly taking 2100 training samples and 450 samples for testing and validation for each review star from 1 to 5. In total, there are 10500 training samples and 2250 each in validation and testing samples.",
        reference="https://uijrt.com/articles/v4/i8/UIJRTV4I80009.pdf",
        dataset={
            "path": "scaredmeow/shopee-reviews-tl-stars",
            "revision": "d096f402fdc76886458c0cfb5dedc829bea2b935",
        },
        type="Classification",
        task_subtypes=["Sentiment/Hate speech"],
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fil-Latn"],
        domains=["Social", "Written"],
        license="mpl-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        date=("2022-05-13", "2023-05-13"),
        main_score="accuracy",
        bibtex_citation="""
        @article{riegoenhancement,
            title={Enhancement to Low-Resource Text Classification via Sequential Transfer Learning},
            author={Riego, Neil Christian R. and Villarba, Danny Bell and Sison, Ariel Antwaun Rolando C. and Pineda, Fernandez C. and Lagunzad, Herminiño C.}
            journal={United International Journal for Research & Technology},
            volume={04},
            issue={08},
            pages={72--82}
        }""",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
