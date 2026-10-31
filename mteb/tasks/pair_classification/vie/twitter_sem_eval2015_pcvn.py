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
@inproceedings{pham2025vnmtebvietnamesemassivetext,
  address = {Rabat, Morocco},
  author = {Pham, Loc and Luu, Tung and Vo, Thu and Nguyen, Minh and Hoang, Viet},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026},
  doi = {10.18653/v1/2026.findings-eacl.86},
  editor = {Demberg, Vera and Inui, Kentaro and Marquez, Llu{\'i}s},
  isbn = {979-8-89176-386-9},
  month = {mar},
  pages = {1705--1725},
  publisher = {Association for Computational Linguistics},
  title = {{VN}-{MTEB}: {V}ietnamese Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2026.findings-eacl.86/},
  year = {2026},
}
""",
        adapted_from=["TwitterSemEval2015"],
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
