import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TriviaQARetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'TriviaQARetrieval',
            'hf_hub_name': 'jinaai/trivia_qa_retrieval',
            'reference': 'https://nlp.cs.washington.edu/triviaqa/',
            'description': (
                'This dataset consists of questions and evidences from the TriviaQA dataset. We only include data from '
                'the filtered version of the test set. Those should contain only evidences that contain a string with '
                'the answer to the question.'
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
