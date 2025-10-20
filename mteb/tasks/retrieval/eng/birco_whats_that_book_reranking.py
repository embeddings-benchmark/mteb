from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BIRCOWhatsThatBookReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-WTB",
        description=(
            "Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query "
            "is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. "
            "The objective is to retrieve the correct book description."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-WTB-Test",
            "revision": "ea824f8065fa62e2e770a8a201c20b03abf87540",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Fiction"],  # Valid domain (Fiction)
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given an ambiguous description of a book, retrieve the book description that best matches the query.",
        bibtex_citation=r"""
@misc{wang2024bircobenchmarkinformationretrieval,
  archiveprefix = {arXiv},
  author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
  eprint = {2402.14151},
  primaryclass = {cs.IR},
  title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  url = {https://arxiv.org/abs/2402.14151},
  year = {2024},
}
""",
    )
