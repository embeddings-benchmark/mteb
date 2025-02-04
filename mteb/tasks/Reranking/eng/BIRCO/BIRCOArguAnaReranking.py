from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from .BIRCOBase import BIRCOBase

class BIRCOArguAnaReranking(BIRCOBase):
    metadata = TaskMetadata(
        name="BIRCOArguAnaReranking",
        description=(
            "BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives.\n\n"
            "ArguAna: In this debate-format IR task, a topic is given and two directly opposing sides of arguments are formed. "
            "A query is an argument that takes one side of the topic, and the objective is to retrieve the passage that takes the "
            "opposing stance and refutes the query's argument."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        dataset={
            "path": "bpHigh/BIRCO_ArguAna",
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
        domains=["Current Affairs"],
        task_subtypes=["Debate Reranking"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{birco2024arguana,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
            author={Wang, Xiaoyue and others},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR}
        }""",
        n_samples={"test": 100},
        avg_character_length={"test": 800}
    )

    def get_query(self, sample):
        instruction = (
            "Instruction: This IR task has a debate format. A query is an argument taking one side of a topic. "
            "Your objective is to retrieve the passage that takes the directly opposing stance, refuting the query’s argument."
        )
        return instruction + "\n" + sample["query"]

    def get_positive_docs(self, sample):
        return sample["positive"]

    def get_negative_docs(self, sample):
        return sample["negative"]
