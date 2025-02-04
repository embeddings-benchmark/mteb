from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from .BIRCOBase import BIRCOBase

class BIRCODorisMaeReranking(BIRCOBase):
    metadata = TaskMetadata(
        name="BIRCODorisMaeReranking",
        description=(
            "BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives.\n\n"
            "DORIS‑MAE: The query consists of users’ needs, leading to several research questions that span a paragraph. "
            "Each candidate passage is an abstract from a scientific paper. The objective is to identify the abstract that most "
            "effectively meets the user's needs."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO/tree/main",
        dataset={
            "path": "bpHigh/BIRCO_Doris_Mae",
            "revision": "b71e754ad70b63fdf902f4f99ca151524c8a80c6"  # Placeholder: update if needed.
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        _needs_graded_evaluation=True,  # Required for graded relevance evaluation.
        main_score="ndcg_at_10",
        date=("2024-02-21", "2020-04-03"),  # Placeholder: update dates if needed.
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Scientific Reranking"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{wang2024birco,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
            author={Xiaoyue Wang and others},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR}
        }""",
        n_samples={"test": 60},
        avg_character_length={"test": 1292.55}
    )

    def get_query(self, sample):
        instruction = (
            "Instruction: The query consists of users’ needs, leading to several research questions that span a paragraph. "
            "Each candidate passage is an abstract from a scientific paper. The objective is to identify the abstract that "
            "most effectively meets the user's needs."
        )
        return instruction + "\n" + sample["query"]

    def get_positive_docs(self, sample):
        return sample["positive"]

    def get_negative_docs(self, sample):
        return sample["negative"]
