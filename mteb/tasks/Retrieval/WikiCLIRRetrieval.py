from collections import defaultdict

import ir_datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikiCLIRRetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'WikiCLIR',
            'ir_datasets_name': 'wikiclir/de',
            'reference': 'https://ir-datasets.com/wikiclir#wikiclir/de',
            'description': (
                'A Cross-Language IR (CLIR) collection between English queries and German documents '
                'built from Wikipedia.'
            ),
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': [self._EVAL_SPLIT],
            'eval_langs': ['en-de'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset = ir_datasets.load(self.description['ir_datasets_name'])

        # load first 1k queries
        queries = defaultdict(dict)
        for item in dataset.queries_iter():
            if len(queries) < 1000:
                queries[item.query_id] = item.first_sent
        # load corpus and qrels
        qrel_dict = defaultdict(dict)
        corpus = {item.doc_id: {'title': item.title, 'text': item.text} for item in dataset.docs_iter()}
        restricted_corpus = defaultdict(dict)
        for item in dataset.qrels_iter():
            if item.query_id in queries.keys():
                qrel_dict[item.query_id][item.doc_id] = item.relevance
                restricted_corpus[item.doc_id] = corpus[item.doc_id]

        self.queries = {self._EVAL_SPLIT: queries}
        self.corpus = {self._EVAL_SPLIT: restricted_corpus}
        self.relevant_docs = {self._EVAL_SPLIT: qrel_dict}

        self.data_loaded = True