from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BIRCODorisMaeReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-DorisMae",
        description=(
            "Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries "
            "that are complex research questions from computer scientists. Each query has a candidate pool of "
            "approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant)."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Retrieval",
        category="t2t",  # MTEB standard category
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-DorisMae-Test",
            "revision": "010fedd0de72e8b77388ea5b1d563d638d5900e5a",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic"],  # Valid combination
        task_subtypes=["Scientific Reranking"],  # MTEB-approved subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify scientific abstracts that fulfill research requirements",
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
