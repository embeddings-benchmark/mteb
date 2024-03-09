import logging
import heapq
from typing import Dict, List, Tuple

import pytrec_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import torch

from .Evaluator import Evaluator
from .utils import cos_sim, dot_score, mrr, recall_cap, hole, top_k_accuracy

logger = logging.getLogger(__name__)

# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class DenseRetrievalExactSearch:    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        # Model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        
        return self.results


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) requires an encode_queries & encode_corpus method.
    This class converts a model with just an .encode method into DRES format.
    """
    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep
        self.use_sbert_model = isinstance(model, SentenceTransformer)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(f"Queries will be truncated to {self.model.get_max_seq_length()} tokens.")
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )
        return self.model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)

def is_dres_compatible(model):
    for method in ["encode_queries", "encode_corpus"]:
        op = getattr(model, method, None)
        if not (callable(op)): return False
    return True


def is_rerank_compatible(model):
    for method in ["rerank"]:
        op = getattr(model, method, None)
        if not (callable(op)): return False
    return True


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class RetrievalEvaluator(Evaluator):
    def __init__(
            self, retriever = None, k_values: List[int] = [1,3,5,10,100,1000], score_function: str = "cos_sim", **kwargs
        ):
        super().__init__(**kwargs)
        if is_dres_compatible(retriever):
            logger.info("The custom encode_queries and encode_corpus functions of the model will be used")
            self.retriever = DenseRetrievalExactSearch(retriever, **kwargs)
        elif is_rerank_compatible(retriever):
            self.retriever = Reranker(retriever, **kwargs)
        else:
            self.retriever = DenseRetrievalExactSearch(DRESModel(retriever), **kwargs)
        self.k_values = k_values
        self.top_k = max(k_values)
        self.score_function = score_function
            
    def __call__(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever: raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)
    
    def rerank(self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int) -> Dict[str, Dict[str, float]]:
    
        new_corpus = {}
    
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]
                    
        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision
    
    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)


# class Reranker:    
#     def __init__(self, model, batch_size: int = 32, **kwargs):
#         # Model is class that provides encode_corpus() and encode_queries()
#         self.model = model
#         self.batch_size = batch_size
#         self.show_progress_bar = kwargs.get("show_progress_bar", True)
#         self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
#         self.results = {}
    
#     def search(self, 
#                corpus: Dict[str, Dict[str, str]], 
#                queries: Dict[str, str], 
#                **kwargs) -> Dict[str, Dict[str, float]]:
#         # Reranks and returns a ranked list with the corpus ids
#         # ignores most other kwargs given to non-cross-encoders
            

#         logger.info("Reranking...")
#         query_ids = list(queries.keys())
#         self.results = {qid: {} for qid in query_ids}
#         queries = [queries[qid] for qid in queries]


#         corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
#         corpus = [corpus[cid] for cid in corpus_ids]


#         pairs = []
#         for q_idx, query in enumerate(queries):
#             for d_idx, doc in enumerate(corpus):
#                 pairs.append((query, doc, (query_ids[q_idx], corpus_ids[d_idx])))

#         logger.info("Reranking in batches... Warning: This might take a while!")
#         itr = range(0, len(pairs), self.batch_size)
        
#         results = {qid: [] for qid in query_ids}  
#         for batch_num, corpus_start_idx in enumerate(itr):
#             logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
#             corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

#             # rerank chunks
#             queries_in_pair = [pair[0] for pair in pairs[corpus_start_idx:corpus_end_idx]]
#             corpus_in_pair = [pair[1] for pair in pairs[corpus_start_idx:corpus_end_idx]]
#             query_ids = [pair[2][0] for pair in pairs[corpus_start_idx:corpus_end_idx]]
#             corpus_ids = [pair[2][1] for pair in pairs[corpus_start_idx:corpus_end_idx]]
#             scores = self.model.rerank(
#                 queries_in_pair, corpus_in_pair,
#             )

#             for i, score in enumerate(scores):
#                 results[query_ids[i]][corpus_ids[i]] = score
        
#         return self.results