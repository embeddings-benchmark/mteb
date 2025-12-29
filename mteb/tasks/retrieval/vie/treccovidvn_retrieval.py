from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TRECCOVIDVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-VN",
        description="A translated dataset from TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        dataset={
            "path": "GreenNode/trec-covid-vn",
            "revision": "54d73a1ea11ea0ae4ec0214ec519c93db79dee88",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
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
        adapted_from=["TRECCOVID"],
    )
