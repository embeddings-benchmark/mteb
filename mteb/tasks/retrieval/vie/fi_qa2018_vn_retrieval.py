from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FiQA2018VN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FiQA2018-VN",
        description="A translated dataset from Financial Opinion Mining and Question Answering The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "GreenNode/fiqa-vn",
            "revision": "6c3cdf6f102151dbbbbc1d2cf999b305eba44dae",
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
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
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
        adapted_from=["FiQA2018"],
    )
