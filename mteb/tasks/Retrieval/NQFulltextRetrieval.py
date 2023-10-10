import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQFulltextRetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'NQFulltextRetrieval',
            'hf_hub_name': 'jinaai/nq_fulltext',
            'reference': 'https://ai.google.com/research/NaturalQuestions',
            'description': (
                'This dataset consists of questions and and wikipedia texts that answer those quesitons.'
                'The original data is from the dev set of the Natural Questions Dataset'
                '(https://ai.google.com/research/NaturalQuestions). The texts in the corpus are derived'
                'from the html source code provided in this datasets. To transform the source code into'
                'plain text beautiful soup (https://pypi.org/project/beautifulsoup4/) was used.'
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

        query_rows = datasets.load_dataset(self.description['hf_hub_name'], 'queries', split=self._EVAL_SPLIT)
        corpus_rows = datasets.load_dataset(self.description['hf_hub_name'], 'corpus', split=self._EVAL_SPLIT)
        qrels_rows = datasets.load_dataset(self.description['hf_hub_name'], 'qrels', split=self._EVAL_SPLIT)

        self.queries = {self._EVAL_SPLIT: {row['_id']: row['text'] for row in query_rows}}
        self.corpus = {self._EVAL_SPLIT: {row['_id']: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {row['_id']: {v: 1 for v in row['text'].split(' ')} for row in qrels_rows}
        }

        self.data_loaded = True
