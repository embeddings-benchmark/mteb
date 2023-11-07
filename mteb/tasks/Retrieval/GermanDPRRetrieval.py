from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask

import datasets


class GermanDPR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _LANGUAGE = "de"

    @property
    def description(self):
        return {
            "name": "GermanDPR",
            "hf_hub_name": "deepset/germandpr",
            "description": "GermanDPR is a German Question Answering dataset for open-domain QA. It associates "
            "questions with a textual context containing the answer",
            "reference": "https://www.deepset.ai/germanquad",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": [_EVAL_SPLIT],
            "eval_langs": [_LANGUAGE],
            "main_score": "ndcg_at_10",
            "revision": "5129d02422a66be600ac89cd3e8531b4f97d347d",
        }

    def _format_documents(self, docs, id_prefix=""):
        result = {}
        for i, (title, content) in enumerate(zip(docs['title'], docs['text'])):
            id_value = f"{id_prefix}{i}"
            formatted_content = content['text'].split('===')[-1].replace('\n', ' ').lstrip()
            result[id_value] = {"title": title, "text": formatted_content}
        return result

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None), split=self._EVAL_SPLIT
        )
        corpus = dict()
        queries = dict()
        relevant_docs = dict()
        for i, row in enumerate(data):
            q_id = f'q_{i}'
            queries[q_id] = row['question']
            pos_docs = self._format_documents(row['positive_ctxs'], id_prefix=f"pos_{i}_")
            neg_docs = self._format_documents(row['hard_negative_ctxs'], id_prefix=f"neg_{i}_")
            corpus.update(pos_docs)
            corpus.update(neg_docs)
            relevant_docs[q_id] = {k: 1 for k in pos_docs}
        self.queries = {self._EVAL_SPLIT: queries}
        self.corpus = {self._EVAL_SPLIT: corpus}
        self.relevant_docs = {self._EVAL_SPLIT: relevant_docs}

        self.data_loaded = True
