from __future__ import annotations


from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class WikiInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="WikiInstructionRetrieval",
        description="Arabic Version of WikiQA by automatic automatic machine translators and crowdsourced the selection of the best one to be incorporated into the corpus.",
        reference="https://huggingface.co/datasets/wiki_qa_ar",
        dataset={
            "path": "wiki_qa_ar",
            "revision": "90f1673b95f7ca68ffc7d8bd1451f8e304819f49",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test"],
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
        n_samples={"eng": 19919 * 2},
        avg_character_length={"eng": 2768.749235474006},
    )
