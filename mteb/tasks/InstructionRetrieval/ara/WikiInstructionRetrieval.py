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

        if "test" not in dataset:
            raise KeyError("Dataset 'test' does not exist in loaded data.")

        # Map answer_id strings to unique integer IDs
        unique_answer_ids = {id: idx for idx, id in enumerate(dataset["test"]["answer_id"])}
        dataset = dataset["test"].map(
            lambda x: {"answer_id_index": unique_answer_ids[x["answer_id"]]}
        )

        dataset = dataset.select(
            [
                "answer_id_index",  # Select based on the new integer index
                "answer_id",       # Keep the original answer_id for reference
                "question",
                "answer",
            ]
        )
        dataset = dataset.rename_column("question", "title")
        dataset = dataset.rename_column("answer", "text")

        # Use the integer index to create mappings
        self.corpus = {
            row["answer_id_index"]: {"text": row["text"], "answer_id": row["answer_id"]}
            for row in dataset
        }
        self.queries = {
            row["answer_id_index"]: row["title"] for row in dataset
        }  # Use the integer index for queries
        self.qrels = {}
        for row in dataset:
            answer_id_index = row["answer_id_index"]
            if answer_id_index not in self.qrels:
                self.qrels[answer_id_index] = {}
            self.qrels[answer_id_index][row["_id"]] = 1
 
