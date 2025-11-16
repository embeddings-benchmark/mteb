import logging
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm

from mteb._create_dataloaders import _create_dataloader_from_texts
from mteb._evaluators.evaluator import Evaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol

logger = logging.getLogger(__name__)


class BitextMiningEvaluator(Evaluator):
    def __init__(
        self,
        sentences: Dataset,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        pair_columns: list[tuple[str, str]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pairs = pair_columns
        self.n = len(sentences)
        self.sentences = sentences
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.task_metadata = task_metadata

    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any]
    ) -> dict[str, list[dict[str, float]]]:
        pair_elements = {p for pair in self.pairs for p in pair}
        if isinstance(self.sentences, Dataset):
            subsets = [
                col for col in self.sentences.features.keys() if col in pair_elements
            ]
        else:
            # BUCC outputs a dict instead of a Dataset
            subsets = list(pair_elements)

        embeddings = {}
        for sub in tqdm(subsets):
            dataloader = _create_dataloader_from_texts(
                self.sentences[sub],
                **encode_kwargs,
            )
            embeddings[sub] = model.encode(
                dataloader,
                task_metadata=self.task_metadata,
                # parallel datasets have lang pairs for subset
                hf_subset=self.hf_subset if self.hf_subset != "parallel" else sub,
                hf_split=self.hf_split,
                **encode_kwargs,
            )

        logger.info("Finding nearest neighbors...")
        neighbours = {}
        for i, (key1, key2) in enumerate(tqdm(self.pairs, desc="Matching sentences")):
            neighbours[f"{key1}-{key2}"] = self._similarity_search(
                embeddings[key1], embeddings[key2], model
            )
        return neighbours

    def _similarity_search(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        model: EncoderProtocol,
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
    ) -> list[dict[str, float]]:
        """This function performs a cosine similarity search between a list of query embeddings and a list of corpus embeddings.

        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

        Args:
            query_embeddings: A 2 dimensional tensor with the query embeddings.
            corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
            model: The model used to encode the queries and corpus. This is used to check if the embeddings are on the same device and to encode the
                queries and corpus if they are not already tensors.
            query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
            corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.

        Returns:
            Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by
                decreasing cosine similarity scores.
        """
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, *query_embeddings.shape)
        if len(corpus_embeddings.shape) == 1:
            corpus_embeddings = corpus_embeddings.reshape(1, *corpus_embeddings)

        # Check that corpus and queries are on the same device
        if (
            isinstance(corpus_embeddings, torch.Tensor)
            and isinstance(query_embeddings, torch.Tensor)
            and corpus_embeddings.device != query_embeddings.device
        ):
            query_embeddings = query_embeddings.to(corpus_embeddings.device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarities
                similarity_scores = model.similarity(  # type: ignore
                    query_embeddings[
                        query_start_idx : query_start_idx + query_chunk_size
                    ],
                    corpus_embeddings[
                        corpus_start_idx : corpus_start_idx + corpus_chunk_size
                    ],
                )

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    torch.tensor(similarity_scores),
                    1,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(similarity_scores)):
                    for sub_corpus_id, score in zip(
                        cos_scores_top_k_idx[query_itr],
                        cos_scores_top_k_values[query_itr],
                    ):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append(
                            {"corpus_id": corpus_id, "score": score}
                        )

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(
                queries_result_list[idx], key=lambda x: x["score"], reverse=True
            )
            queries_result_list[idx] = queries_result_list[idx][0]

        return queries_result_list
