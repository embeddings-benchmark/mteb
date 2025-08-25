from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
	AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SwedishPatentCPCSubclassClassification(AbsTaskMultilabelClassification):
	metadata = TaskMetadata(
		name="SwedishPatentCPCSubclassClassification",
		description="""This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system. Each document can have multiple labels, making this a multi-label classification task with significant implications for patent retrieval and prior art search. 
		The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.""",
		reference="https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254",
		type="MultilabelClassification",
		category="s2s",
		modalities=["text"],
		eval_splits=["train"],
		eval_langs=["swe"],
		main_score="accuracy",
		dataset = {
			"path": "atheer2104/swedish-patent-cpc-subclass",
			"revision": "cf7b50cc195c609f2cb8a0f6b6651335794eccb4",
		},
		date = ("2025-01-01", "2025-04-01"),
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
		abstract = {Patents are essential for protecting intellectual property and advancing innovation, but the accessibility of historical patents is often limited by outdated classification systems. In Sweden, many older patents are classified solely under the now-obsolete German Patent Classification (DPK) system, making them difficult to retrieve using the modern Cooperative Patent Classification (CPC) system. This lack of accessibility can hinder prior art searches and public access to technical knowledge. The thesis investigates whether incorporating DPK information alongside patent claims improves the performance of machine learning models in classifying historical Swedish patents into the CPC system. Three models - Linear, XML-CNN, and CNN-BiLSTM - were trained on four datasets: two using only patent claims and two combining patent claims with DPK information for both CPC subclass and group-level predictions. Model performance was evaluated using precision at k, recall at k, F1-score at k, normalized discounted cumulated gains (nDCG@k) for k = 1, 3, 5, and micro F1-score. The results show that incorporating DPK information consistently enhances classification performance across all metrics and models, with statistical significance improvements. While group-level predictions were more challenging due to label imbalance and fewer samples per label, DPK information improved results at both hierarchy levels. These findings demonstrate that leveraging legacy classification data can significantly improve the reclassification of historical patents, thereby enhancing their accessibility and supporting more effective prior art searches. },
		year = {2025},
		url = "https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254"
		}
		""")
	
	def dataset_transform(self):
		self.dataset = self.stratified_subsampling(
			self.dataset, seed=self.seed, splits=["train"], n_samples=8192
		)