from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DBPediaVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-VN",
        description="A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "GreenNode/dbpedia-vn",
            "revision": "c3e20179fbcee16217ef9461a14a54b7faca9b63",
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
        domains=["Written", "Encyclopaedic"],
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
        adapted_from=["DBPedia"],
    )
