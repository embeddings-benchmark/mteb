from datasets import load_dataset

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class News21InstructionRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="News21InstructionRetrieval",
        description="Measuring retrieval instruction following ability on News21 narratives for the FollowIR benchmark.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/news21-instructions-mteb",
            "revision": "39db677749b3b783bb277d0e2d4712f5f133f52b",
        },
        type="InstructionReranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
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
        qrel_diff_ds = load_dataset(
            self.metadata.dataset["path"],
            "qrel_diff",
            split="qrel_diff",
            revision=self.metadata.dataset["revision"],
        )
        changed_qrels = {item["query-id"]: item["corpus-ids"] for item in qrel_diff_ds}

        return evaluate_p_mrr_change(
            qrels,
            results,
            changed_qrels,
            self.k_values,
        )
