import datasets

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "NarrativeQARetrieval",
            "hf_hub_name": "narrativeqa",
            "reference": "https://metatext.io/datasets/narrativeqa",
            "description": (
                "NarrativeQA is a dataset for the task of question answering on long narratives. It consists of "
                "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], split=self._EVAL_SPLIT
        )
        self.queries = {
            self._EVAL_SPLIT: {
                str(i): row["question"]["text"] for i, row in enumerate(data)
            }
        }
        self.corpus = {
            self._EVAL_SPLIT: {
                str(row["document"]["id"]): {"text": row["document"]["text"]}
                for row in data
            }
        }
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                str(i): {row["document"]["id"]: 1} for i, row in enumerate(data)
            }
        }

        self.data_loaded = True
