from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from .BIRCOBase import BIRCOBase

class BIRCORELICReranking(BIRCOBase):
    metadata = TaskMetadata(
        name="BIRCORELICReranking",
        description=(
            "BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives.\n\n"
            "RELIC: In this task, the query is a piece of literary analysis with one quotation intentionally masked (replaced with [masked sentence(s)]). "
            "The objective is to retrieve the passage that naturally fills the masked portion without repeating query content."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        dataset={
            "path": "bpHigh/BIRCO_RELIC",
            "revision": "YOUR_REVISION_PLACEHOLDER"  # Placeholder: update with actual revision.
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        _needs_graded_evaluation=True,
        main_score="ndcg_at_10",
        date=("2024-04-03", "2024-04-03"),  # Placeholder: update dates if needed.
        form=["written"],
        domains=["Literature"],
        task_subtypes=["Literary Analysis"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{birco2024relic,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
            author={Wang, Xiaoyue and others},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR}
        }""",
        n_samples={"test": 100},
        avg_character_length={"test": 850}
    )

    def get_query(self, sample):
        instruction = (
            "Instruction: In this task, the query is a piece of scholarly literary analysis where one quotation has been masked "
            "(by [masked sentence(s)]). The objective is to retrieve the passage that naturally and appropriately fills the masked portion."
        )
        return instruction + "\n" + sample["query"]

    def get_positive_docs(self, sample):
        return sample["positive"]

    def get_negative_docs(self, sample):
        return sample["negative"]
