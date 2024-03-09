import logging
import heapq
from typing import Dict, List, Tuple
import pandas as pd
import tqdm

import pytrec_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import torch
from collections import defaultdict

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
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = defaultdict(list)

    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               instructions: Dict[str, str],
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        
        # reranking here, so no need for the heap
        top_k = len(corpus)
        qid = list(queries.keys())[0] 
        assert len(queries) == 1
        
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
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, instructions=instructions)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        results = {qid: [] for qid in query_ids}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            if self.save_corpus_embeddings and "qid" in kwargs and len(self.corpus_embeddings[kwargs["qid"]]):
                sub_corpus_embeddings = self.corpus_embeddings[qid][batch_num]
            else:
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor = self.convert_to_tensor
                )
                if self.save_corpus_embeddings and "qid" in kwargs:
                    self.corpus_embeddings[kwargs["qid"]].append(sub_corpus_embeddings)

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # get all values to put in results
            cos_scores = cos_scores.cpu().tolist()
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in enumerate(cos_scores[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    results[query_id].append((corpus_id, score))


        for qid in results:
            for doc_id, score in results[qid]:
                self.results[qid][doc_id] = score
        
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
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = {}

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
        if "qid" in kwargs and self.save_corpus_embeddings and len(self.corpus_embeddings) > 0:
            return self.corpus_embeddings[kwargs["qid"]]
        
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
        corpus_embeddings = self.model.encode(sentences, batch_size=batch_size, **kwargs)
        if self.save_corpus_embeddings and "qid" in kwargs:
            self.corpus_embeddings[kwargs["qid"]] = corpus_embeddings.cpu().detach()
        return corpus_embeddings
    


class Reranker:    
    def __init__(self, model, batch_size: int = 32, **kwargs):
        # Model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        logger.info("Reranker initialized with batch size: {}".format(batch_size))
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
        self.sep = kwargs.get("sep", " ")



    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               instructions: Dict[str, str],
               top_k, score_function, # all ignored
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Reranks and returns a ranked list with the corpus ids
        # ignores most other kwargs given to non-cross-encoders
            

        logger.info("Reranking...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]
        
        corpus = [
            (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
            for doc in corpus
        ]

        pairs = []
        for q_idx, query in enumerate(queries):
            for d_idx, doc in enumerate(corpus):
                pairs.append((query, doc, instructions[query], (query_ids[q_idx], corpus_ids[d_idx])))

        logger.info("Reranking in batches... Warning: This might take a while!")
        itr = range(0, len(pairs), self.batch_size)
        
        results = {qid: {} for qid in query_ids}  
        for batch_num, corpus_start_idx in enumerate(tqdm.tqdm(itr, leave=False)):
            # logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.batch_size, len(corpus))

            # rerank chunks
            queries_in_pair = [pair[0] for pair in pairs[corpus_start_idx:corpus_end_idx]]
            corpus_in_pair = [pair[1] for pair in pairs[corpus_start_idx:corpus_end_idx]]        

            instructions_in_pair = [pair[2] for pair in pairs[corpus_start_idx:corpus_end_idx]]
            query_ids = [pair[3][0] for pair in pairs[corpus_start_idx:corpus_end_idx]]
            corpus_ids = [pair[3][1] for pair in pairs[corpus_start_idx:corpus_end_idx]]
            assert len(queries_in_pair) == len(corpus_in_pair) == len(instructions_in_pair)
            scores = self.model.rerank(
                queries_in_pair, corpus_in_pair, instructions=instructions_in_pair
            )

            for i, score in enumerate(scores):
                results[query_ids[i]][corpus_ids[i]] = score
        
        return results
    

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
class InstructionRetrievalEvaluator(Evaluator):
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
            
    def __call__(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], instructions: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever: raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, instructions, self.top_k, self.score_function, **kwargs)
    
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



    @staticmethod 
    def get_rank_from_dict(dict_of_results, doc_id):
        tuple_of_id_score = dict_of_results.items()
        # sort them by score
        sorted_by_score = sorted(tuple_of_id_score, key=lambda x: x[1], reverse=True)
        # return the rank of the doc_id, if not found return -1
        for i, (id, score) in enumerate(sorted_by_score):
            if id == doc_id:
                return i + 1, score
            
        return len(sorted_by_score) + 1, 0



    @staticmethod
    def evaluate_change(original_run, new_run, changed_qrels):
        changes = []
        for qid in changed_qrels.keys():
            original_qid_run = original_run[qid]
            new_qid_run = new_run[qid]
            for idx, changed_doc in enumerate(changed_qrels[qid]):
                original_rank, original_score = InstructionRetrievalEvaluator.get_rank_from_dict(original_qid_run, changed_doc)
                new_rank, new_score = InstructionRetrievalEvaluator.get_rank_from_dict(new_qid_run, changed_doc)
                change = int(original_rank - new_rank)
                changes.append(
                    {
                        "qid": qid,
                        "doc_id": changed_doc,
                        "change": change,
                        "relevance": 0,
                        "og_rank": original_rank,
                        "new_rank": new_rank,
                        "og_score": original_score,
                        "new_score": new_score
                    }
                )


        # we now have a DF of [qid, doc_id, change] to run our calculations with
        changes_df = pd.DataFrame(changes)
        changes_df["rankwise_score"] = changes_df.apply(lambda x: InstructionRetrievalEvaluator.rank_score(x), axis=1)
        changes_df["pointwise_score"] = changes_df.apply(lambda x: InstructionRetrievalEvaluator.pointwise_score(x), axis=1)

        doc_wise = changes_df.groupby("doc_id").agg({"rankwise_score": "mean", "pointwise_score": "mean"})

        # do qid wise calculations
        qid_wise = changes_df.groupby("qid").agg({"rankwise_score": "mean", "pointwise_score": "mean"})

        return {
            "per_doc": doc_wise.to_dict(orient="index"),
            "per_qid": qid_wise.to_dict(orient="index"),
            "rankwise_score": qid_wise["rankwise_score"].mean(),
            "pointwise_score": qid_wise["pointwise_score"].mean()
        }

    @staticmethod
    def rank_score(x: dict):
        # if x["og_rank"] == 0 and x["new_rank"] == 0:
        #     return 0

        if x["og_rank"] >= x["new_rank"]:
            return ((1/x["og_rank"]) / (1/x["new_rank"])) - 1
        else:
            return (1 - ((1/x["new_rank"]) / (1/x["og_rank"])))

    @staticmethod
    def pointwise_score(x: dict):
        if x["og_score"] == 0 and x["new_score"] == 0:
            return 0

        if x["og_score"] >= x["new_score"]:
            return (x["og_score"] - x["new_score"]) / x["og_score"]
        else:
            return -1 * ((x["new_score"] - x["og_score"]) / x["new_score"])


