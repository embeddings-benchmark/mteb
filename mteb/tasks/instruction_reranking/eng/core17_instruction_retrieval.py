from datasets import load_dataset

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Core17InstructionRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Core17InstructionRetrieval",
        description="Measuring retrieval instruction following ability on Core17 narratives for the FollowIR benchmark.",
        reference="https://aclanthology.org/2025.naacl-long.597/",
        dataset={
            "path": "mteb/Core17InstructionRetrieval",
            "revision": "67d8733dd0691f08f34ed9dda5941578cba8d91c",
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
@inproceedings{weller2024followir,
  address = {Albuquerque, New Mexico},
  author = {Weller, Orion and Chang, Benjamin and MacAvaney, Sean and Lo, Kyle and Cohan, Arman and Van Durme, Benjamin and Lawrie, Dawn and Soldaini, Luca},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the {A}mericas Chapter of the {A}ssociation for {C}omputational {L}inguistics: Human Language Technologies (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.naacl-long.597},
  editor = {Chiruzzo, Luis and Ritter, Alan and Wang, Lu},
  isbn = {979-8-89176-189-6},
  month = {apr},
  pages = {11926--11942},
  publisher = {Association for Computational Linguistics},
  title = {{F}ollow{IR}: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  url = {https://aclanthology.org/2025.naacl-long.597/},
  year = {2025},
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
