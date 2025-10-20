from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BIRCOArguAnaReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-ArguAna",
        description=(
            "Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries "
            "and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument "
            "that directly refutes the query’s stance."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ArguAna-Test",
            "revision": "0229bee36500c92028874b55b5a1d08e5ee37bb9",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Written"],  # there is no 'Debate' domain
        task_subtypes=["Reasoning as Retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given a one-paragraph argument, retrieve the passage that contains the counter-argument which directly refutes the query's stance.",
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
