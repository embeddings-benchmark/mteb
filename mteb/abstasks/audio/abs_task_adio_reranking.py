import logging
from collections import defaultdict
from copy import copy

import datasets
from datasets import Audio, Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

logger = logging.getLogger(__name__)

OLD_FORMAT_RERANKING_TASKS = [
    "AudioCapsMiniReranking",
    "ESC50AudioReranking",
    "FSDnoisy18kAudioReranking",
    "GTZANAudioReranking",
    "UrbanSound8KAudioReranking",
    "VocalSoundAudioReranking",
]


class AbsTaskAudioReranking(AbsTaskRetrieval):
    """Reranking task class.

    Warning: Deprecated
        This class is deprecated and will be removed in future versions. Please use the updated retrieval tasks instead.
        You can add your task name to mteb.abstasks.text.reranking.OLD_FORMAT_RERANKING_TASKS to load it in new format.
        You can reupload it using `task.push_dataset_to_hub('your/repository')` after loading the data.
        For dataformat and other information, see [AbsTaskRetrieval][mteb.abstasks.retrieval.AbsTaskRetrieval].
    """

    def load_data(self) -> None:
        """Load the dataset."""
        if self.data_loaded:
            return

        if self.metadata.name in OLD_FORMAT_RERANKING_TASKS:
            self.transform_old_dataset_format()
            self.dataset_transform()
            return
        else:
            # use AbsTaskRetrieval default to load the data
            return super().load_data()

    def _process_example(self, example: dict, split: str, query_idx: int) -> dict:
        """Process a single example from the dataset.

        Args:
            example: A single example from the dataset containing 'query', 'positive', and 'negative' fields.
            split: The dataset split (e.g., 'train', 'validation', 'test').
            query_idx: The index of the query in the dataset split.

        Returns:
            A dictionary containing the processed example with query_id, query text, document ids, document texts, and relevance scores.
        """
        query = example["query"]
        positive_docs = example["positive"]
        negative_docs = example["negative"]

        query_id = f"{split}_query{query_idx}"

        # Initialize the structures for this example
        example_data = {
            "query_id": query_id,
            "query": query,
            "doc_ids": [],
            "doc_audio": [],
            "relevance_scores": [],
        }

        for i, pos_audio in enumerate(positive_docs):
            # format i as a five digit number
            formatted_i = str(i).zfill(5)
            # have "a" in front so that positives are first, then negatives
            #   this shouldn't matter except for ties, and the previous reranking results
            #   had the positives first
            doc_id = f"apositive_{query_id}_{formatted_i}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_audio"].append(pos_audio)
            example_data["relevance_scores"].append(1)

        for i, neg_audio in enumerate(negative_docs):
            formatted_i = str(i).zfill(5)
            doc_id = f"negative_{query_id}_{formatted_i}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_audio"].append(neg_audio)
            example_data["relevance_scores"].append(0)

        return example_data

    def transform_old_dataset_format(self, given_dataset: Dataset | None = None):
        """Transform the old format to the new format using HF datasets mapping. This is a one-time transformation for datasets which are in the old format.

        Args:
            given_dataset (Dataset, optional): The dataset to transform. Defaults to None. This is helpful for some older datasets which are loaded with custom code, but need to be transformed still.
        """
        if self.metadata.name not in OLD_FORMAT_RERANKING_TASKS:
            return

        logging.info(
            f"Transforming old format to standard format for {self.metadata.name}"
        )

        given_dataset = copy(given_dataset)
        self.dataset = defaultdict(lambda: defaultdict(dict))

        hf_subsets = self.hf_subsets

        for hf_subset in hf_subsets:
            if given_dataset:
                cur_dataset = given_dataset
                if hf_subset in cur_dataset:
                    cur_dataset = cur_dataset[hf_subset]
            elif "name" in self.metadata.dataset:
                cur_dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
                assert hf_subset == "default", (
                    f"Only default subset is supported for {self.metadata.name} since `name` is given in the metadata."
                )
            else:
                cur_dataset = datasets.load_dataset(
                    **self.metadata.dataset, name=hf_subset
                )  # type: ignore

            for split in cur_dataset:
                corpus = []
                queries = []
                relevant_docs = defaultdict(dict)
                top_ranked = defaultdict(list)

                # Create an enumerated dataset to pass indices
                enumerated_dataset = Dataset.from_dict(
                    {
                        "index": range(len(cur_dataset[split])),
                        "query": cur_dataset[split]["query"],
                        "positive": cur_dataset[split]["positive"],
                        "negative": cur_dataset[split]["negative"],
                    }
                )

                # first, filter out the ones that have no positive or no negatives
                enumerated_dataset = enumerated_dataset.filter(
                    lambda example: len(example["positive"]) > 0
                    and len(example["negative"]) > 0
                )

                logger.info(
                    f"Filtered out {len(cur_dataset[split]) - len(enumerated_dataset)} examples with no positive or no negative examples. {len(enumerated_dataset)} examples remaining."
                )

                # Map the transformation function over the dataset
                processed_dataset = enumerated_dataset.map(
                    lambda example, idx: self._process_example(example, split, idx),
                    with_indices=True,
                    remove_columns=enumerated_dataset.column_names,
                )

                # Populate the data structures
                for item in processed_dataset:
                    query_id = item["query_id"]
                    queries.append({"id": query_id, "audio": item["query"]})

                    # Add documents and relevance information
                    for doc_id, doc_audio, relevance in zip(
                        item["doc_ids"], item["doc_audio"], item["relevance_scores"]
                    ):
                        corpus.append(
                            {
                                "audio": doc_audio,
                                "id": doc_id,
                            }
                        )
                        top_ranked[query_id].append(doc_id)
                        relevant_docs[query_id][doc_id] = relevance

                corpus = Dataset.from_list(corpus)
                corpus = corpus.cast_column(
                    "audio",
                    Audio(),
                )
                queries = Dataset.from_list(queries)
                queries = queries.cast_column(
                    "audio",
                    Audio(),
                )

                self.dataset[hf_subset][split] = RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=relevant_docs,
                    top_ranked=top_ranked,
                )
        self.data_loaded = True
