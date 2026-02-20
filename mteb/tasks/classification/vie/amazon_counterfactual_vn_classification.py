from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonCounterfactualVNClassification(AbsTaskClassification):
    num_samples = 32

    metadata = TaskMetadata(
        name="AmazonCounterfactualVNClassification",
        dataset={
            "path": "GreenNode/amazon-counterfactual-vn",
            "revision": "b48bc27d383cfca5b6a47135a52390fa5f66b253",
        },
        description="A collection of translated Amazon customer reviews annotated for counterfactual detection pair classification. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://arxiv.org/abs/2104.06893",
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
        domains=["Reviews", "Written"],
        task_subtypes=["Counterfactual Detection"],
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
        adapted_from=["AmazonCounterfactualClassification"],
    )
