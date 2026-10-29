from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TweetSentimentExtractionVNClassification(AbsTaskClassification):
    num_samples = 32

    metadata = TaskMetadata(
        name="TweetSentimentExtractionVNClassification",
        description="A collection of translated tweets annotated for sentiment extraction. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "GreenNode/tweet-sentiment-extraction-vn",
            "revision": "f453803eff1e91579eb235dc1d7c38d39b3f1340",
        },
        type="Classification",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
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
        adapted_from=["TweetSentimentExtractionClassification"],
    )
