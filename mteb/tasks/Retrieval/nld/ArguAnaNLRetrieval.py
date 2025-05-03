from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaNL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAna-NL",
        description="ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is "
        "a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-arguana",
        dataset={
            "path": "clips/beir-nl-arguana",
            "revision": "4cd085d148fe2cac923bb7758d6ef585926170ba",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2016-03-01", "2016-03-01"),  # best guess: based on publication date
        domains=["Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
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
        adapted_from=["ArguAna"],
    )
