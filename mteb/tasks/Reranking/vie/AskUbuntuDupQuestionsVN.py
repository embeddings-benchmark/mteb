from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class AskUbuntuDupQuestionsVN(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AskUbuntuDupQuestions-VN",
        description="""A translated dataset from AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.""",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "GreenNode/askubuntudupquestions-reranking-vn",
            "revision": "5cfaa5c07252d30c37302bfc056f0d85884971a1",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="map",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Programming", "Web"],
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
        adapted_from=["AskUbuntuDupQuestions"],
    )
