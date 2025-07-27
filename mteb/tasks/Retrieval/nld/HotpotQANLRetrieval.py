from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQANL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-NL",
        dataset={
            "path": "clips/beir-nl-hotpotqa",
            "revision": "3ab9eaff3e1d6f7f81914ae3b2bdd1e8511cf9a8",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            + "supervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is "
            "a Dutch translation. "
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
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
        adapted_from=["HotpotQA"],
    )
