from __future__ import annotations

from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from mteb import (
    AbsTaskBitextMining,
    AbsTaskClassification,
    AbsTaskClustering,
    AbsTaskClusteringFast,
    AbsTaskPairClassification,
    AbsTaskRetrieval,
    AbsTaskSTS,
    AbsTaskSummarization,
)
from mteb.abstasks.AbsTask import AbsTask


def upload_task_to_hf(task: AbsTask, repo_name: str) -> None:
    if isinstance(task, AbsTaskBitextMining):
        pass
    elif isinstance(task, AbsTaskClassification):
        pass
    elif isinstance(task, AbsTaskClustering) or isinstance(task, AbsTaskClusteringFast):
        pass
    elif isinstance(task, AbsTaskPairClassification):
        pass
    elif isinstance(task, AbsTaskRetrieval):
        upload_retrieval(task, repo_name)
    elif isinstance(task, AbsTaskSTS):
        pass
    elif isinstance(task, AbsTaskSummarization):
        pass


def upload_retrieval(task: AbsTaskRetrieval, repo_name: str) -> None:
    if task.data_loaded is False:
        task.load_data()
    if task.is_multilingual:
        for config in tqdm(task.queries):
            queries_dataset = {}
            for split in task.queries[config]:
                # print(split)
                # print(task.queries[config][split])
                queries_dataset[split] = Dataset.from_list(
                    [
                        {
                            "_id": idx,
                            "text": text,
                        }
                        for idx, text in task.queries[config][split].items()
                    ]
                )
            queries_dataset = DatasetDict(queries_dataset)
            queries_dataset.push_to_hub(repo_name, f"{config}-queries")

            corpus_dataset = {}
            for split in task.corpus[config]:
                corpus_dataset[split] = Dataset.from_list(
                    [
                        {
                            "_id": idx,
                            "text": text
                            if isinstance(text, str)
                            else (
                                text.get("title", "") + " " if text.get("title") else ""
                            )
                            + text["text"],
                            "title": "",
                        }
                        for idx, text in task.corpus[config][split].items()
                    ]
                )
            corpus_dataset = DatasetDict(corpus_dataset)
            corpus_dataset.push_to_hub(repo_name, f"{config}-corpus")

            relevant_docs_dataset = {}
            for split in task.relevant_docs[config]:
                relevant_docs_dataset[split] = Dataset.from_list(
                    [
                        {"query-id": query_id, "corpus-id": doc_id, "score": score}
                        for query_id, docs in task.relevant_docs[config][split].items()
                        for doc_id, score in docs.items()
                    ]
                )
            relevant_docs_dataset = DatasetDict(relevant_docs_dataset)
            relevant_docs_dataset.push_to_hub(repo_name, f"{config}-qrels")

            if task.instructions:
                instructions_dataset = {}
                for split in task.instructions[config]:
                    instructions_dataset[split] = Dataset.from_list(
                        [
                            {
                                "query-id": idx,
                                "text": text,
                            }
                            for idx, text in task.instructions[config][split].items()
                        ]
                    )
                instructions_dataset = DatasetDict(instructions_dataset)
                instructions_dataset.push_to_hub(repo_name, f"{config}-instructions")
            if task.top_ranked:
                top_ranked_dataset = {}
                for split in task.top_ranked[config]:
                    top_ranked_dataset[split] = Dataset.from_list(
                        [
                            {
                                "query-id": query_id,
                                "corpus-ids": doc_id,
                            }
                            for query_id, docs in task.top_ranked[config][split].items()
                            for doc_id in docs  # todo add case if docs is dict
                        ]
                    )
                top_ranked_dataset = DatasetDict(top_ranked_dataset)
                top_ranked_dataset.push_to_hub(repo_name, f"{config}-top_ranked")
                # DatasetDict(
                #     {split: Dataset.from_dict(task.top_ranked[config][split]) for split in task.top_ranked[config]}
                # ).push_to_hub(repo_name, f"{config}-top_ranked")
    else:
        DatasetDict(
            {
                split: Dataset.from_list(
                    [
                        {
                            "_id": idx,
                            "text": text,
                        }
                        for idx, text in task.queries[split].items()
                    ]
                )
                for split in task.queries
            }
        ).push_to_hub(repo_name, "queries")
        DatasetDict(
            {
                split: Dataset.from_list(
                    [
                        {"_id": idx, "text": text, "title": ""}
                        for idx, text in task.corpus[split].items()
                    ]
                )
                for split in task.corpus
            }
        ).push_to_hub(repo_name, "corpus")
        DatasetDict(
            {
                split: Dataset.from_list(
                    [
                        {"query-id": query_id, "corpus-id": doc_id, "score": score}
                        for query_id, docs in task.relevant_docs[split].items()
                        for doc_id, score in docs.items()
                    ]
                )
                for split in task.relevant_docs
            }
        ).push_to_hub(repo_name, "default")
        if task.instructions:
            DatasetDict(
                {
                    split: Dataset.from_list(
                        [
                            {
                                "query-id": idx,
                                "text": text,
                            }
                            for idx, text in task.instructions[split].items()
                        ]
                    )
                    for split in task.instructions
                }
            ).push_to_hub(repo_name, "instruction")
        if task.top_ranked:
            DatasetDict(
                {
                    split: Dataset.from_list(
                        [
                            {
                                "query-id": query_id,
                                "corpus-ids": doc_id,
                            }
                            for query_id, docs in task.top_ranked[split].items()
                            for doc_id in docs  # todo add case if dict
                        ]
                    )
                    for split in task.top_ranked
                }
            ).push_to_hub(repo_name, "top_ranked")


if __name__ == "__main__":
    import mteb
    for task in ["T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval", ]:
        task1 = mteb.get_task(task)
        upload_task_to_hf(task1, f"mteb/{task}")
    # task1 = mteb.get_task("AppsRetrieval")
    # upload_task_to_hf(task1, "AppsRetrieval")
