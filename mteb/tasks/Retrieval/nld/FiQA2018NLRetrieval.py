from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQA2018NL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018-NL",
        description="Financial Opinion Mining and Question Answering. FiQA2018-NL is a Dutch translation",
        reference="https://huggingface.co/datasets/clips/beir-nl-fiqa",
        dataset={
            "path": "clips/beir-nl-fiqa",
            "revision": "a66648414dbbcd03e3eda02988c428be78097fd0",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2018-04-01", "2018-04-01"),  # best guess: based on publication date
        domains=["Written", "Non-fiction"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
        adapted_from=["FiQA2018"],
    )
