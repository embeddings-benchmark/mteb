from collections import defaultdict

import ir_datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCODocumentRetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'dev'

    @property
    def description(self):
        return {
            'name': 'MSMARCODocumentRetrieval',
            'ir_datasets_name': 'msmarco-document/dev',
            'reference': 'https://microsoft.github.io/msmarco/',
            'description': (
                'MS MARCO is a collection of datasets focused on deep learning in search. It consists of datasets for '
                'passage ranking and full document ranking. This task concerns the full document ranking dataset.'
            ),
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': [self._EVAL_SPLIT],
            'eval_langs': ['en'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset = ir_datasets.load(self.description['ir_datasets_name'])

        # load queries
        queries = {item.query_id: item.text for item in dataset.queries_iter()}
        # load corpus
        corpus = {item.doc_id: {'title': item.title, 'text': item.body} for item in dataset.docs_iter()}
        # load qrels
        qrel_dict = defaultdict(dict)
        for item in dataset.qrels_iter():
            qrel_dict[item.query_id][item.doc_id] = item.relevance

        self.queries = {self._EVAL_SPLIT: queries}
        self.corpus = {self._EVAL_SPLIT: corpus}
        self.relevant_docs = {self._EVAL_SPLIT: qrel_dict}

        self.data_loaded = True
