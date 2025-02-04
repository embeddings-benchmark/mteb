from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from .BIRCOBase import BIRCOBase

class BIRCOWhatsThatBookReranking(BIRCOBase):
    metadata = TaskMetadata(
        name="BIRCOWhatsThatBookReranking",
        description=(
            "BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives.\n\n"
            "WhatsThatBook: In this task, a user is trying to recall the name of a book based on ambiguous details (places, events, characters, etc.). "
            "The objective is to retrieve the book description or summary that best matches the query details."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        dataset={
            "path": "bpHigh/BIRCO_WhatsThatBook",
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
        task_subtypes=["Book Retrieval"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{birco2024whatsthatbook,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
            author={Wang, Xiaoyue and others},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR}
        }""",
        n_samples={"test": 100},
        avg_character_length={"test": 700}
    )

    def get_query(self, sample):
        instruction = (
            "Instruction: In this task, the query describes a book using vague and informal details. "
            "Your objective is to retrieve the book description or summary that best matches the details given in the query."
        )
        return instruction + "\n" + sample["query"]

    def get_positive_docs(self, sample):
        return sample["positive"]

    def get_negative_docs(self, sample):
        return sample["negative"]
