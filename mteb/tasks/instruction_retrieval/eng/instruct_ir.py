from mteb._evaluators.retrieval_metrics import robustness_at_10
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class InstructIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InstructIR",
        description='A benchmark specifically designed to evaluate the instruction following ability in information retrieval models. Our approach focuses on user-aligned instructions tailored to each query instance, reflecting the diverse characteristics inherent in real-world search scenarios. **NOTE**: scores on this may differ unless you include instruction first, then "[SEP]" and then the query via redefining `combine_query_and_instruction` in your model.',
        reference="https://github.com/kaistAI/InstructIR/tree/main",
        dataset={
            "path": "mteb/InstructIR-mteb",
            "revision": "42c3afabe480643b755a7099dbf0f9ebeedaf6ca",
        },
        type="InstructionRetrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="robustness_at_10",
        date=("2024-02-05", "2024-02-06"),
        domains=["Web"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{oh2024instructir,
  archiveprefix = {{arXiv}},
  author = {{Hanseok Oh and Hyunji Lee and Seonghyeon Ye and Haebin Shin and Hansol Jang and Changwook Jun and Minjoon Seo}},
  eprint = {{2402.14334}},
  primaryclass = {{cs.CL}},
  title = {{INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models}},
  year = {{2024}},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return {"robustness_at_10": robustness_at_10(qrels, results, scores)}
