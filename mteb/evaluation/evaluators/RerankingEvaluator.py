from .Evaluator import Evaluator
import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score
from .utils import cos_sim

class RerankingEvaluator(Evaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, mrr_at_k: int = 10, name: str = '', similarity_fct=cos_sim, batch_size: int = 64, use_batched_encoding: bool = True):
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.use_batched_encoding = use_batched_encoding

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [sample for sample in self.samples if len(sample['positive']) > 0 and len(sample['negative']) > 0]

    def __call__(self, model):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model):
        return self.compute_metrics_batched(model) if self.use_batched_encoding else self.compute_metrics_individual(model)

    def compute_metrics_batched(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents together
        """
        all_mrr_scores = []
        all_ap_scores = []

        all_query_embs = model.encode([sample['query'] for sample in self.samples],
                                  convert_to_tensor=True,
                                  batch_size=self.batch_size)

        all_docs = []

        for sample in self.samples:
            all_docs.extend(sample['positive'])
            all_docs.extend(sample['negative'])

        all_docs_embs = model.encode(all_docs,
                                    convert_to_tensor=True,
                                    batch_size=self.batch_size)

        #Compute scores
        query_idx, docs_idx = 0,0
        for instance in self.samples:
            query_emb = all_query_embs[query_idx]
            query_idx += 1

            num_pos = len(instance['positive'])
            num_neg = len(instance['negative'])
            docs_emb = all_docs_embs[docs_idx:docs_idx+num_pos+num_neg]
            docs_idx += num_pos+num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  #Sort in decreasing order

            #Compute MRR score
            is_relevant = [True]*num_pos + [False]*num_neg
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores.cpu().tolist()))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {'map': mean_ap, 'mrr': mean_mrr}


    def compute_metrics_individual(self, model):
        """
        Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        all_mrr_scores = []
        all_ap_scores = []


        for instance in tqdm.tqdm(self.samples, desc="Samples"):
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            query_emb = model.encode([query], convert_to_tensor=True, batch_size=self.batch_size)
            docs_emb = model.encode(docs, convert_to_tensor=True, batch_size=self.batch_size)

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  #Sort in decreasing order

            #Compute MRR score
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores.cpu().tolist()))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {'map': mean_ap, 'mrr': mean_mrr}