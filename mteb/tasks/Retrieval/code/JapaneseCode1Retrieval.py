from __future__ import annotations

import logging
from pathlib import Path

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class JapaneseCode1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JapaneseCode1Retrieval",
        description="Japanese code retrieval dataset. Japanese natural language queries paired with Python code snippets for cross-lingual code retrieval evaluation.",
        reference="https://huggingface.co/datasets/mteb-private/JapaneseCode1Retrieval",
        dataset={
            "path": "local",
            "revision": "1.0.0",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train", "test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )

    def load_data(self, **kwargs):
        """Load data from local directory."""
        if self.data_loaded:
            return

        # Path to the local dataset
        dataset_path = Path(
            "/Users/fodizoltan/Projects/toptal/voyageai/tmp/rteb/data/JapaneseCode1Retrieval"
        )

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            logger.info(f"Loading {split} split...")

            # Load the dataset from parquet files
            import pandas as pd

            split_path = dataset_path / f"{split}.parquet"

            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")

            df = pd.read_parquet(split_path)

            # Initialize split dictionaries
            corpus_split = {}
            queries_split = {}
            qrels_split = {}

            # Process each row in the dataframe
            for _, row in df.iterrows():
                qid = row["qid"]
                query_text = row["query"]
                positive_docs = row["positive"] if row["positive"] is not None else []

                # Add query
                queries_split[qid] = query_text

                # Add documents to corpus and create qrels
                qrels_split[qid] = {}

                for doc in positive_docs:
                    doc_id = doc["docid"]
                    doc_text = doc["text"]
                    corpus_split[doc_id] = doc_text
                    qrels_split[qid][doc_id] = 1

            # Store the processed data
            self.corpus[split] = corpus_split
            self.queries[split] = queries_split
            self.relevant_docs[split] = qrels_split

            logger.info(
                f"Loaded {len(queries_split)} queries and {len(corpus_split)} documents for {split} split"
            )

        self.data_loaded = True
