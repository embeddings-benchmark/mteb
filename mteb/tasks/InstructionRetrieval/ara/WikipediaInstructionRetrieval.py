from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class WikipediaInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="WikipediaInstructionRetrieval",
        description="This dataset contains a pre-processed version from Wikipedia suitable for semantic search.",
        reference="https://huggingface.co/datasets/Cohere/wikipedia-22-12-ar-embeddings",
        dataset={
            "path": "Cohere/wikipedia-22-12-ar-embeddings",
            "revision": "ea5f00014bd7626aa55affb07de57d519ab3309a",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["ara-Arap"],
        main_score="p-MRR",
        date=("2023-01-14", "2024-03-22"),
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation=""" """,
        n_samples={"ara": 3113764},
        avg_character_length={"ara": 2768.749235474006},
    )
   
