from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
	AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SwedishPatentCPCGroupClassification(AbsTaskMultilabelClassification):
	metadata = TaskMetadata(
		name="SwedishPatentCPCGroupClassification",
		description="""This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system at the group level. Each document can have multiple labels, making this a challenging multi-label classification task with significant class imbalance and data sparsity characteristics. The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.""",
		reference="https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254",
		type="MultilabelClassification",
		category="s2s",
		modalities=["text"],
		eval_splits=["train"],
		eval_langs=["swe-Latn"],
		main_score="accuracy",
		dataset = {
			"path": "atheer2104/swedish-patent-cpc-group",
			"revision": "a898492d9f808ddc63862e49db4fa969ed8497f1",
		},
		date = ("1885-01-01", "1972-01-01"),
		domains = ["Legal", "Government"],
		task_subtypes = [],
		license = "mit",
		annotations_creators="expert-annotated",
		dialect=[],
		sample_creation="found",
		bibtex_citation="""
		@mastersthesis{Salim1987995,
		author = {Salim, Atheer},
		institution = {KTH, School of Electrical Engineering and Computer Science (EECS)},
		pages = {70},
		school = {KTH, School of Electrical Engineering and Computer Science (EECS)},
		title = {Machine Learning for Classifying Historical Swedish Patents : A Comparison of Textual and Combined Data Approaches},
		series = {TRITA-EECS-EX},
		number = {2025:571},
		keywords = {Multi-label Text Classification, Machine Learning, Patent Classification, Deep Learning, Natural Language Processing, Textklassificering med flera Klasser, Maskininlärning, Patentklassificering, Djupinlärning, Språkteknologi},
		year = {2025},
		url = "https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254"
		}
		""")
	
	def dataset_transform(self):
		self.dataset = self.stratified_subsampling(
			self.dataset, seed=self.seed, splits=["train"], n_samples=8192
		)