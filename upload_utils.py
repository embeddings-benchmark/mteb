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
    if not task.data_loaded:
        task.load_data()

    if isinstance(task, AbsTaskBitextMining):
        upload_bitext_mining(task, repo_name)
    elif isinstance(task, AbsTaskClassification):
        upload_classification(task, repo_name)
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


def upload_bitext_mining(task: AbsTaskBitextMining, repo_name: str) -> None:
    if task.is_multilingual:
        for hf_subset in task.metadata.eval_langs:
            sentences = {}
            if task.parallel_subsets:
                for split in task.dataset:
                    sent_1, sent_2 = hf_subset.split("-")
                    sentences[split] = Dataset.from_dict(
                        {
                            "sentence1": task.dataset[split][sent_1],
                            "sentence2": task.dataset[split][sent_2],
                        }
                    )
            else:
                sent_1, sent_2 = task.get_pairs(task.parallel_subsets)[0]
                for split in task.dataset[hf_subset]:
                    sentences[split] = Dataset.from_dict(
                        {
                            "sentence1": task.dataset[hf_subset][split][sent_1],
                            "sentence2": task.dataset[hf_subset][split][sent_2],
                        }
                    )
            sentences = DatasetDict(sentences)
            sentences.push_to_hub(repo_name, hf_subset)
    else:
        sentences = {}
        for split in task.dataset:
            sent_1, sent_2 = task.get_pairs(task.parallel_subsets)[0]
            sentences[split] = Dataset.from_dict(
                {
                    "sentence1": task.dataset[split][sent_1],
                    "sentence2": task.dataset[split][sent_2],
                }
            )
        sentences = DatasetDict(sentences)
        sentences.push_to_hub(repo_name)


def upload_classification(task: AbsTaskClassification, repo_name: str) -> None:
    if task.is_multilingual:
        for hf_subset in task.metadata.eval_langs:
            sentences = {}
            for split in task.dataset[hf_subset]:
                print(hf_subset, split)
                sentences[split] = Dataset.from_dict(
                    {
                        "text": task.dataset[hf_subset][split]["text"],
                        "label": task.dataset[hf_subset][split]["label"],
                    }
                )
            sentences = DatasetDict(sentences)
            sentences.push_to_hub(repo_name, hf_subset)
    else:
        sentences = {}
        for split in task.dataset:
            sentences[split] = Dataset.from_dict(
                {
                    "text": task.dataset[split]["text"],
                    "label": task.dataset[split]["label"],
                }
            )
        sentences = DatasetDict(sentences)
        sentences.push_to_hub(repo_name)


def upload_retrieval(task: AbsTaskRetrieval, repo_name: str) -> None:
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
        queries_dataset = {}
        if "default" in task.queries:
            # old rerankers have additional default split
            task.queries = task.queries["default"]
            task.corpus = task.corpus["default"]
            task.relevant_docs = task.relevant_docs["default"]
            if task.instructions:
                task.instructions = task.instructions["default"]
            if task.top_ranked:
                task.top_ranked = task.top_ranked["default"]
        for split in task.queries:
            queries_dataset[split] = Dataset.from_list(
                [
                    {
                        "_id": idx,
                        "text": text,
                    }
                    for idx, text in task.queries[split].items()
                ]
            )
        queries_dataset = DatasetDict(queries_dataset)
        queries_dataset.push_to_hub(repo_name, "queries")
        corpus_dataset = {}
        for split in task.corpus:
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
                    for idx, text in task.corpus[split].items()
                ]
            )
        corpus_dataset = DatasetDict(corpus_dataset)
        corpus_dataset.push_to_hub(repo_name, "corpus")
        relevant_docs_dataset = {}
        for split in task.relevant_docs:
            print(task.relevant_docs[split])
            relevant_docs_dataset[split] = Dataset.from_list(
                [
                    {"query-id": query_id, "corpus-id": doc_id, "score": score}
                    for query_id, docs in task.relevant_docs[split].items()
                    for doc_id, score in docs.items()
                ]
            )
        relevant_docs_dataset = DatasetDict(relevant_docs_dataset)
        relevant_docs_dataset.push_to_hub(repo_name, "default")
        if task.instructions:
            instructions_dataset = {}
            for split in task.instructions:
                instructions_dataset[split] = Dataset.from_list(
                    [
                        {
                            "query-id": idx,
                            "text": text,
                        }
                        for idx, text in task.instructions[split].items()
                    ]
                )
            instructions_dataset = DatasetDict(instructions_dataset)
            instructions_dataset.push_to_hub(repo_name, "instructions")
        if task.top_ranked:
            top_ranked_dataset = {}
            for split in task.top_ranked:
                print(task.top_ranked[split])
                top_ranked_dataset[split] = Dataset.from_list(
                    [
                        {
                            "query-id": query_id,
                            "corpus-ids": docs,
                        }
                        for query_id, docs in task.top_ranked[split].items()
                    ]
                )
            top_ranked_dataset = DatasetDict(top_ranked_dataset)
            top_ranked_dataset.push_to_hub(repo_name, "top_ranked")


if __name__ == "__main__":
    import mteb

    for task in [
        # "T2Retrieval",
        # "MMarcoRetrieval",
        # "DuRetrieval",
        # "CovidRetrieval",
        # "CmedqaRetrieval",
        # "EcomRetrieval",
        # "MedicalRetrieval",
        # "VideoRetrieval",
        # "WikipediaRetrievalMultilingual",
        # "AskUbuntuDupQuestions", # TODO
    ]:
        task1 = mteb.get_task(task)
        # upload_task_to_hf(task1, task)
        upload_task_to_hf(task1, f"mteb/{task}")
        # model = mteb.get_model("intfloat/multilingual-e5-small")
        # evaluator = mteb.MTEB([task1])
        # evaluator.run(model, overwrite_results=True)
