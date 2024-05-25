from __future__ import annotations
import random

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from datasets import load_dataset

TEST_SAMPLES = 1000 

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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        random.seed(42)
        split = self.metadata_dict["eval_splits"][0]  # Assuming 'test' split

        # Load the dataset with the 'plain_text' configuration
        ds = load_dataset(**self.metadata_dict["dataset"], split=split, name='plain_text').shuffle(seed=42)

        # Determine the relevance key based on the split
        relevance_key = "label"  # Default to "label" for train/validation

        # Extract relevant data and filter out irrelevant pairs
        data = [(row["question_id"], row["question"], row["document_id"], row["answer"], row[relevance_key]) 
                for row in ds if row[relevance_key] == '1']  # Only keep relevant pairs

        data = random.sample(data, min(TEST_SAMPLES, len(data)))

        self.queries = {split: {}}
        self.corpus = {split: {}}
        self.relevant_docs = {split: {}}

        document2id = {}
        answer2id = {}

        for i, (question_id, question, document_id, answer, _) in enumerate(data):
            self.queries[split][question_id] = question

            # Add document to corpus if it hasn't been seen before
            if document_id not in document2id:
                document2id[document_id] = len(self.corpus[split])
                self.corpus[split][str(document2id[document_id])] = {"id": document_id}  # Placeholder for document text

            # Add answer to corpus if it hasn't been seen before
            if answer not in answer2id:
                answer2id[answer] = len(self.corpus[split])
                self.corpus[split][str(answer2id[answer])] = {"id": answer}  # Placeholder for answer text

            self.relevant_docs[split].setdefault(question_id, {})[str(answer2id[answer])] = 1  # All documents equally relevant

        self.data_loaded = True

