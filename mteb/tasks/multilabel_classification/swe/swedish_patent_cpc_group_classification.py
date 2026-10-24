from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class SwedishPatentCPCGroupClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SwedishPatentCPCGroupClassification",
        description="This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system at the group level. Each document can have multiple labels, making this a challenging multi-label classification task with significant class imbalance and data sparsity characteristics. The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.",
        reference="https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254",
        type="MultilabelClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        dataset={
            "path": "atheer2104/swedish-patent-cpc-group-new",
            "revision": "d1980d69e2fcf11e912025ba6bb1e3afe6b9168a",
        },
        date=("1885-01-01", "1972-01-01"),
        domains=["Legal", "Government"],
        task_subtypes=[],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], n_samples=8192
        )

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
