from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from datasets import load_dataset

class WikiInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="WikiInstructionRetrieval",
        description="Arabic Version of WikiQA by automatic automatic machine translators and crowdsourced the selection of the best one to be incorporated into the corpus.",
        reference="https://huggingface.co/datasets/wiki_qa_ar",
        dataset={
            "path": "wiki_qa_ar",
            "revision": "90f1673b95f7ca68ffc7d8bd1451f8e304819f49",
            "name": "plain_text",
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
        n_samples={"ara": 19919 * 2},
        avg_character_length={"ara": 2768.749235474006},
    )

    def load_data(self, **kwargs):
        dataset = load_dataset(
            self.metadata.dataset["path"],
            name=self.metadata.dataset["name"],
            revision=self.metadata.dataset["revision"],
            trust_remote_code=True,
        )


        dataset = dataset.map(
            lambda x: {
                "question": x["question"],
                "answer": x["answer"],
            }
        )

        self.corpus = {
            idx: {"text": row["answer"]}
            for idx, row in enumerate(dataset)
        }
        self.queries = {
            idx: row["question"] for idx, row in enumerate(dataset)
        }  # Use the incremented ID for queries
        self.qrels = {}
        for idx, row in enumerate(dataset):
            if idx not in self.qrels:
                self.qrels[idx] = {}
            self.qrels[idx][idx] = 1
