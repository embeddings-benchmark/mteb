from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from datasets import load_dataset

TEST_SAMPLES = 2000 

class WikiInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="WikiInstructionRetrieval",
        description="Arabic Version of WikiQA by automatic automatic machine translators and crowdsourced the selection of the best one to be incorporated into the corpus.",
        reference="https://huggingface.co/datasets/Ruqiya/wiki_qa_ar/",
        dataset={
            "path": "Ruqiya/wiki_qa_ar",
            "revision": "0a0730791bd24d8196acbbfcf23190b8de4ec0a8",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test", "validation", "train"],
        eval_langs=["ara-Arab"],
        main_score="p-MRR",
        date=("2023-01-14", "2024-03-22"),
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=""" """,
        n_samples={"ara": TEST_SAMPLES},
        avg_character_length={"ara": 2768.749235474006},
    )


