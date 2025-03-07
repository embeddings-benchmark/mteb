from __future__ import annotations

from typing import Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.encoder_interface import BatchedInput, Conversation


def create_dataloader_from_texts(
    text: list[str], **dataloader_kwargs
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def corpus_to_dict(
    corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
) -> dict[str, list[str | None]]:
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + " " + corpus["text"][i]).strip()
            if "title" in corpus
            else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]
        titles = corpus["title"]
        bodies = corpus["text"]
    elif isinstance(corpus, list) and isinstance(corpus[0], dict):
        sentences = [
            (doc["title"] + " " + doc["text"]).strip()
            if "title" in doc
            else doc["text"].strip()
            for doc in corpus
        ]
        titles = [doc.get("title", "") for doc in corpus]
        bodies = [doc.get("text", "") for doc in corpus]
    else:
        sentences = corpus
        titles = [""] * len(corpus)
        bodies = [""] * len(corpus)
    return {
        "text": sentences,
        "title": titles,
        "body": bodies,
    }


def create_dataloader_for_retrieval_corpus(
    inputs: list[dict[str, str]] | dict[str, list[str]] | list[str], **dataloader_kwargs
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a corpus.

    Args:
        inputs: Corpus
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the corpus.
    """
    dataset = Dataset.from_dict(corpus_to_dict(inputs))
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def create_dataloader_for_queries(
    queries: list[str],
    instructions: list[str] | None = None,
    combine_query_and_instruction: Callable[[str, str], str] | None = None,
    **dataloader_kwargs,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        instructions: A list of instructions. If None, the dataloader will only contain the queries.
        combine_query_and_instruction: A function that combines a query with an instruction. If None, the default function will be used.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the queries.
    """
    if instructions is None:
        dataset = Dataset.from_dict({"text": queries, "query": queries})
    else:
        dataset = Dataset.from_dict(
            {
                "text": [
                    combine_query_and_instruction(q, i)
                    for q, i in zip(queries, instructions)
                ],
                "instruction": instructions,
                "query": queries,
            }
        )
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def convert_conv_history_to_query(
    conversations: list[list[str | Conversation]],
) -> list[str]:
    conversations_converted = []

    for conversation in conversations:
        # if it's a list of strings, just join them
        if isinstance(conversation[0], str):
            conv_str = "; ".join(conversation)
        # otherwise, it's a list of dictionaries, which we need to convert to strings
        elif isinstance(conversation[0], dict):
            conv = []
            for i, turn in enumerate(conversation):
                error_msg = (
                    "When converting conversations lists of dictionary to string, each turn in the conversation "
                    + "must be a dictionary with 'role' and 'content' keys"
                )
                if not isinstance(turn, dict):
                    raise ValueError(f"Turn {i} is not a dictionary. " + error_msg)

                # check for keys 'role' and 'content' in the dictionary, if not found, raise an error
                if "role" not in turn:
                    raise ValueError(
                        "Key 'role' not found in the dictionary. " + error_msg
                    )
                if "content" not in turn:
                    raise ValueError(
                        "Key 'content' not found in the dictionary. " + error_msg
                    )

                conv.append(f"{turn['role']}: {turn['content']}")
            conv_str = "; ".join(conv)
        else:
            raise ValueError(
                "Conversations must be a list consisting of strings or dictionaries with 'role' and 'content' keys"
            )

        conversations_converted.append(conv_str)

    return conversations_converted


def create_dataloader_for_queries_conversation(
    queries: list[list[str | Conversation]],
    instructions: list[str] | None = None,
    combine_query_and_instruction: Callable[[str, str], str] | None = None,
    **dataloader_kwargs,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        instructions: A list of instructions. If None, the dataloader will only contain the queries.
        combine_query_and_instruction: A function that combines a query with an instruction. If None, the default function will be used.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the queries.
    """
    converted_queries = convert_conv_history_to_query(queries)
    if instructions is None:
        dataset = Dataset.from_dict({"text": converted_queries, "query": queries})
    else:
        dataset = Dataset.from_dict(
            {
                "text": [
                    combine_query_and_instruction(q, i)
                    for q, i in zip(converted_queries, instructions)
                ],
                "instruction": instructions,
                "query": queries,
            }
        )
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
