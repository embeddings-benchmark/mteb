import uuid
from typing import Dict, List

import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HagridRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["dev"]

    @property
    def description(self):
        return {
            "name": "HagridRetrieval",
            "hf_hub_name": "miracl/hagrid",
            "reference": "https://github.com/project-miracl/hagrid",
            "description": (
                "HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)"
                "is a dataset for generative information-seeking scenarios. It consists of queries"
                "along with a set of manually labelled relevant passages"
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": self._EVAL_SPLITS,
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "b2a085913606be3c4f2f1a8bff1810e38bade8fa",
        }

    def load_data(self, **kwargs):
        """
        Loads the different split of the dataset (queries/corpus/relevants)
        """
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            "miracl/hagrid", split=self._EVAL_SPLITS[0], revision=self.description.get("revision", None)
        )
        proc_data = self.preprocess_data(data)

        self.queries = {self._EVAL_SPLITS[0]: {d["query_id"]: d["query_text"] for d in proc_data}}
        self.corpus = {self._EVAL_SPLITS[0]: {d["answer_id"]: {"text": d["answer_text"]} for d in proc_data}}
        self.relevant_docs = {self._EVAL_SPLITS[0]: {d["query_id"]: {d["answer_id"]: 1} for d in proc_data}}

        self.data_loaded = True

    def preprocess_data(self, dataset: Dict) -> List[Dict]:
        """
        Preprocessed the data in a format easirer
        to handle for the loading of queries and corpus
        ------
        PARAMS
        dataset : the hagrid dataset (json)
        """

        preprocessed_data = []
        for d in dataset:
            # get the best answer among positively rated answers
            best_answer = self.get_best_answer(d)
            # if no good answer found, skip
            if best_answer is not None:
                preprocessed_data.append(
                    {
                        "query_id": str(d["query_id"]),
                        "query_text": d["query"],
                        "answer_id": str(uuid.uuid4()),
                        "answer_text": best_answer,
                    }
                )

        return preprocessed_data

    def get_best_answer(self, data: Dict) -> str:
        """
        Get the best answer among available answers
        of a query.
        WARNING : May return None if no good answer available
        --------
        PARAMS:
        data: a dict representing one element of the dataset
        """
        good_answers = [a["answer"] for a in data["answers"] if a["informative"] == 1 and a["attributable"] == 1]
        # Return 1st one if >=1 good answers else None
        return good_answers[0] if len(good_answers) > 0 else None
