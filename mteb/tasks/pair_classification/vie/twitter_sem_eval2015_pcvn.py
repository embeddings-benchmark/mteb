from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TwitterSemEval2015PCVN(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterSemEval2015-VN",
        dataset={
            "path": "GreenNode/twittersemeval2015-pairclassification-vn",
            "revision": "9215a3c954078fd15c2bbecca914477d53944de1",
        },
        description="A translated dataset from Paraphrase-Pairs of Tweets from the SemEval 2015 workshop. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://alt.qcri.org/semeval2015/task1/",
        category="t2c",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="max_ap",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Social", "Written"],
        task_subtypes=[],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["TwitterSemEval2015"],
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
