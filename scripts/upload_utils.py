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
)
from mteb.abstasks.AbsTask import AbsTask


def upload_task_to_hf(task: AbsTask, repo_name: str) -> None:
    """
    Uploads a task's dataset to Hugging Face hub based on its type.

    Args:
        task (AbsTask): The task object.
        repo_name (str): The name of the repository on Hugging Face hub.
    """
    if not task.data_loaded:
        task.load_data()

    task_upload_mapping = {
        AbsTaskBitextMining: upload_bitext_mining,
        AbsTaskClassification: upload_classification,
        AbsTaskClustering: upload_clustering,
        AbsTaskClusteringFast: upload_clustering,
        AbsTaskPairClassification: upload_pair_classification,
        AbsTaskSTS: upload_sts,
        AbsTaskRetrieval: upload_retrieval,
    }

    for task_type, upload_fn in task_upload_mapping.items():
        if isinstance(task, task_type):
            upload_fn(task, repo_name)
            return

    raise NotImplementedError(f"Task type {type(task)} not implemented.")


def push_to_hub(repo_name: str, sentences: dict, subset_name: str = None) -> None:
    """
    Pushes a dataset to the Hugging Face hub.

    Args:
        repo_name (str): The name of the repository on Hugging Face hub.
        sentences (dict): A dictionary with splits and corresponding datasets.
        subset_name (str, optional): The name of the subset (used for multilingual datasets).
    """
    sentences = DatasetDict(sentences)
    if subset_name:
        sentences.push_to_hub(repo_name, subset_name, commit_message=f"Add {subset_name} dataset")
    else:
        sentences.push_to_hub(repo_name, commit_message="Add dataset")


def process_dataset(task: AbsTask, repo_name: str, fields: list[str], is_multilingual: bool) -> None:
    """
    Generic function to process datasets for multilingual and non-multilingual tasks.

    Args:
        task (AbsTask): The task object.
        repo_name (str): The name of the repository on Hugging Face hub.
        fields (list[str]): List of field names to extract from the dataset.
        is_multilingual (bool): Whether the task is multilingual.
    """
    if is_multilingual:
        for subset in task.metadata.eval_langs:
            sentences = {}
            for split in task.dataset[subset]:
                sentences[split] = Dataset.from_dict(
                    {field: task.dataset[subset][split][field] for field in fields}
                )
            push_to_hub(repo_name, sentences, subset)
    else:
        sentences = {}
        for split in task.dataset:
            sentences[split] = Dataset.from_dict(
                {field: task.dataset[split][field] for field in fields}
            )
        push_to_hub(repo_name, sentences)


def upload_classification(task: AbsTaskClassification, repo_name: str) -> None:
    process_dataset(task, repo_name, ["text", "label"], task.is_multilingual)


def upload_clustering(task: AbsTaskClustering | AbsTaskClusteringFast, repo_name: str) -> None:
    process_dataset(task, repo_name, ["sentences", "labels"], task.is_multilingual)


def upload_pair_classification(task: AbsTaskPairClassification, repo_name: str) -> None:
    process_dataset(task, repo_name, ["sentence1", "sentence2", "labels"], task.is_multilingual)


def upload_sts(task: AbsTaskSTS, repo_name: str) -> None:
    process_dataset(task, repo_name, ["sentence1", "sentence2", "score"], task.is_multilingual)


def upload_bitext_mining(task: AbsTaskBitextMining, repo_name: str) -> None:
    """
    Uploads a Bitext Mining task dataset to Hugging Face hub.

    Args:
        task (AbsTaskBitextMining): The task object.
        repo_name (str): The name of the repository on Hugging Face hub.
    """
    if task.is_multilingual:
        for hf_subset in task.metadata.eval_langs:
            sentences = {}
            if task.parallel_subsets:
                # If there are parallel subsets, process them
                for split in task.dataset:
                    sent_1, sent_2 = hf_subset.split("-")
                    sentences[split] = Dataset.from_dict(
                        {
                            "sentence1": task.dataset[split][sent_1],
                            "sentence2": task.dataset[split][sent_2],
                        }
                    )
            else:
                # Handle the non-parallel subset case
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


def upload_retrieval(task: AbsTaskRetrieval, repo_name: str) -> None:
    def format_text_field(text):
        """Formats the text field to match loader expectations."""
        if isinstance(text, str):
            return text
        return f"{text.get('title', '')} {text.get('text', '')}".strip()

    if task.is_multilingual:
        for config in tqdm(task.queries):
            queries_dataset = {}
            for split in task.queries[config]:
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
                            "text": format_text_field(text),
                            "title": text.get("title", "") if isinstance(text, dict) else "",
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
                                "instruction": text,
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
                                "corpus-ids": docs,
                            }
                            for query_id, docs in task.top_ranked[config][split].items()
                        ]
                    )
                top_ranked_dataset = DatasetDict(top_ranked_dataset)
                top_ranked_dataset.push_to_hub(repo_name, f"{config}-top_ranked")
    else:
        if "default" in task.queries:
            # old rerankers have additional default split
            task.queries = task.queries["default"]
            task.corpus = task.corpus["default"]
            task.relevant_docs = task.relevant_docs["default"]
            if task.instructions:
                task.instructions = task.instructions["default"]
            if task.top_ranked:
                task.top_ranked = task.top_ranked["default"]

        queries_dataset = {}
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
                        "text": format_text_field(text),
                        "title": text.get("title", "") if isinstance(text, dict) else "",
                    }
                    for idx, text in task.corpus[split].items()
                ]
            )

        corpus_dataset = DatasetDict(corpus_dataset)
        corpus_dataset.push_to_hub(repo_name, "corpus")
        relevant_docs_dataset = {}
        for split in task.relevant_docs:
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
                            "instruction": text,
                        }
                        for idx, text in task.instructions[split].items()
                    ]
                )
            instructions_dataset = DatasetDict(instructions_dataset)
            instructions_dataset.push_to_hub(repo_name, "instructions")
        if task.top_ranked:
            top_ranked_dataset = {}
            for split in task.top_ranked:
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
        # retrieval
        "T2Retrieval",
        "MMarcoRetrieval",
        "DuRetrieval",
        "CovidRetrieval",
        "CmedqaRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "VideoRetrieval",
        "WikipediaRetrievalMultilingual",
        # reranking
        "AskUbuntuDupQuestions",
        # instruct retrieval
        "mFollowIR",
        # bitext mining
        "IWSLT2017BitextMining",
        # classification
        "AmazonReviewsClassification",
        # clustering
        "IndicReviewsClusteringP2P",
        # pairclassification
        "XNLIV2",
        # sts
        "IndicCrosslingualSTS"
    ]:
        print(task)
        task1 = mteb.get_task(task)
        upload_task_to_hf(task1, f"mteb/{task}")
        # model = mteb.get_model("intfloat/multilingual-e5-small")
        # evaluator = mteb.MTEB([task1])
        # evaluator.run(model, overwrite_results=True)
