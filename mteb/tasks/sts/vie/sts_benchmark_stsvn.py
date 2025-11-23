from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STSBenchmarkSTSVN(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSBenchmark-VN",
        dataset={
            "path": "GreenNode/stsbenchmark-sts-vn",
            "revision": "f24d66738cda4a02138ada5af7689a92ce1fcad6",
        },
        description="A translated dataset from Semantic Textual Similarity Benchmark (STSbenchmark) dataset. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="cosine_spearman",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Blog", "News", "Written"],
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
        adapted_from=["STSBenchmark"],
    )
    min_score = 0
    max_score = 5
