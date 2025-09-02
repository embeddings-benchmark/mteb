from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CQADupstackWordpressVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWordpress-VN",
        dataset={
            "path": "GreenNode/cqadupstack-wordpress-vn",
            "revision": "2230f80e1baf42aa005731ca86577621c566fcd7",
        },
        description="""A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.""",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Written", "Web", "Programming"],
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
        adapted_from=["CQADupstackWordpress"],
    )
