from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class SwedishPatentCPCSubclassClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SwedishPatentCPCSubclassClassification",
        description="This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system. Each document can have multiple labels, making this a multi-label classification task with significant implications for patent retrieval and prior art search. The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.",
        reference="https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254",
        type="MultilabelClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        dataset={
            "path": "atheer2104/swedish-patent-cpc-subclass-new",
            "revision": "114fcab0a716a27cf3f54a7ebd6e08f45f62de88",
        },
        date=("1885-01-01", "1972-01-01"),
        domains=["Legal", "Government"],
        task_subtypes=[],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{Salim1987995,
  author = {Salim, Atheer},
  institution = {KTH, School of Electrical Engineering and Computer Science (EECS)},
  keywords = {Multi-label Text Classification, Machine Learning, Patent Classification, Deep Learning, Natural Language Processing, Textklassificering med flera Klasser, Maskininlärning, Patentklassificering, Djupinlärning, Språkteknologi},
  number = {2025:571},
  pages = {70},
  school = {KTH, School of Electrical Engineering and Computer Science (EECS)},
  series = {TRITA-EECS-EX},
  title = {Machine Learning for Classifying Historical Swedish Patents : A Comparison of Textual and Combined Data Approaches},
  url = {https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254},
  year = {2025},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], n_samples=8192
        )

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
