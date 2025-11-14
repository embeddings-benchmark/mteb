from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MassiveScenarioVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MassiveScenarioVNClassification",
        dataset={
            "path": "GreenNode/amazon-massive-scenario-vn",
            "revision": "a82e282d9f5aec1a8cf7d868ce40f70669c16b89",
        },
        description="A translated dataset from MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.",
        category="t2c",
        type="Classification",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Spoken"],
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
        adapted_from=["MassiveScenarioClassification"],
    )
