from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SciDocsRerankingVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciDocsRR-VN",
        description="A translated dataset from Ranking of related scientific papers based on their title. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "mteb/SciDocsRR-VN",
            "revision": "760894958694fbd1559ceef44505abe5b59ea598",
        },
        type="Reranking",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="map_at_1000",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
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
        adapted_from=["SciDocsRR"],
    )
