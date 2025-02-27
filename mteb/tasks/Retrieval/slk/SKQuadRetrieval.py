from __future__ import annotations
from datasets import load_dataset
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

class SKQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SKQuadRetrieval",
        description=(
            "Retrieval SK Quad evaluates Slovak search performance using questions and answers "
            "derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers "
            "based on their relevancy to corresponding questions, which is vital for improving "
            "Slovak language search systems."
        ),
        reference="https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad",
        dataset={
            "path": "TUKE-KEMT/retrieval-skquad",
            "revision": "09f81f51dd5b8497da16d02c69c98d5cb5993ef2",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2024-05-30", "2024-06-13"),
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_dataset_and_validate(self, dataset_name, dataset_config, expected_columns, trust_remote_code=False):
        
        try:
            dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=trust_remote_code)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None

        
        for split in dataset.keys():
            for column in expected_columns:
                if column not in dataset[split].column_names:
                    print(f"Expected column '{column}' not found in {split} split")
                    return self.preprocess_data(dataset, expected_columns)

        return dataset

    def preprocess_data(self, dataset, expected_columns):
        
        processed_data = {}
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            for column in expected_columns:
                if column not in df.columns:
                    df[column] = None  

            processed_data[split] = df.to_dict(orient="records")

        print("Data preprocessed successfully.")
        return processed_data

    def load_data(self, eval_splits=None, **kwargs):
        """Load and preprocess datasets for retrieval task."""
        eval_splits = eval_splits or ["test"]

        # Expected columns for validation
        expected_columns_default = ["query-id", "corpus-id", "score"]
        expected_columns_corpus = ["_id", "text", "title"]
        expected_columns_queries = ["_id", "text"]

        # Load and validate datasets
        ds_default = self.load_dataset_and_validate("TUKE-KEMT/retrieval-skquad", "default", expected_columns_default)
        ds_corpus = self.load_dataset_and_validate("TUKE-KEMT/retrieval-skquad", "corpus", expected_columns_corpus)
        ds_query = self.load_dataset_and_validate("TUKE-KEMT/retrieval-skquad", "queries", expected_columns_queries)

        if "test" in eval_splits:
            self.corpus = {
                "test": {
                    row["_id"]: {"text": row["text"], "title": row["title"]}
                    for row in ds_corpus["corpus"]
                }
            }
            self.queries = {
                "test": {row["_id"]: row["text"] for row in ds_query["queries"]}
            }
            self.relevant_docs = {"test": {}}

            for row in ds_default["test"]:
                self.relevant_docs["test"].setdefault(row["query-id"], {})[
                    row["corpus-id"]
                ] = int(row["score"])

            print(
                f"Data Loaded:\n- Corpus size: {len(self.corpus['test'])}\n- Query size: {len(self.queries['test'])}\n- Relevance entries: {len(self.relevant_docs['test'])}"
            )

def main():
    # Initialize the SKQuadRetrieval class
    task = SKQuadRetrieval()
    # Load data
    task.load_data()

if __name__ == "__main__":
    main()
