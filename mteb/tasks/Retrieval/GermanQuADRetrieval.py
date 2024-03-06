from collections import defaultdict
from datasets import load_dataset, DatasetDict

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(hf_hub_name, 'corpus')
    queries_dataset = load_dataset(hf_hub_name, 'queries')
    qrels = load_dataset(hf_hub_name + '-qrels')[eval_split]

    corpus = {e['_id']: {'text': e['text']} for e in corpus_dataset['corpus']}
    queries = {e['_id']: e['text'] for e in queries_dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['query-id']][e['corpus-id']] = e['score']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs

class GermanQuADRetrieval(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": "GermanQuAD-Retrieval",
            "hf_hub_name": "mteb/germanquad-retrieval",
            "description": "Context Retrieval for German Question Answering",
            "reference": "https://www.deepset.ai/germanquad",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "mrr_at_10",
            "revision": "f5c87ae5a2e7a5106606314eef45255f03151bb3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True
