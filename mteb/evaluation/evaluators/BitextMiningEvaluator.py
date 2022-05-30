import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .utils import cos_sim

from .Evaluator import Evaluator


class BitextMiningEvaluator(Evaluator):
    def __init__(self, sentences1, sentences2, gold, batch_size=32):
        self.gold = gold
        self.sentences1 = [sentences1[i] for (i, j) in self.gold]
        self.sentences2 = sentences2

        self.batch_size = batch_size

    def __call__(self, model):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model):
        # Compute embeddings
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = np.asarray([emb_dict[sent] for sent in self.sentences1])
        embeddings2 = np.asarray([emb_dict[sent] for sent in self.sentences2])

        # Find nearest neighbors
        nearest_neighbors = self._similarity_search(embeddings1, embeddings2, top_k=1)

        # Compute errors
        errors = 0
        labels = []
        predictions = []
        for i, x in enumerate(nearest_neighbors):
            j = x[0]["corpus_id"]
            labels.append(j)
            predictions.append(self.gold[i][1])
        
        scores = {
            "precision": precision_score(labels, predictions, average='weighted'),
            "recall": recall_score(labels, predictions, average='weighted'),
            "f1": f1_score(labels, predictions, average='weighted'),
            "accuracy": accuracy_score(labels, predictions),
        }
        return scores

    def _similarity_search(
        self,
        query_embeddings,
        corpus_embeddings,
        query_chunk_size=100,
        corpus_chunk_size=500000,
        top_k=10,
        score_function=cos_sim,
    ):
        """
        This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param query_embeddings: A 2 dimensional tensor with the query embeddings.
        :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
        :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
        :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
        :param top_k: Retrieve top k matching entries.
        :param score_function: Function for computing scores. By default, cosine similarity.
        :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
        """
        query_embeddings = torch.from_numpy(query_embeddings)
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        if len(corpus_embeddings.shape) == 1:
            corpus_embeddings = corpus_embeddings.unsqueeze(0)

        # Check that corpus and queries are on the same device
        if corpus_embeddings.device != query_embeddings.device:
            query_embeddings = query_embeddings.to(corpus_embeddings.device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarities
                cos_scores = score_function(
                    query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                    corpus_embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
                )

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
                )
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(
                        cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                    ):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append({"corpus_id": corpus_id, "score": score})

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x["score"], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list
