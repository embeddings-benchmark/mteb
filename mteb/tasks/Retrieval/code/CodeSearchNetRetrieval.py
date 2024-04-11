from __future__ import annotations
import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="CodeSearchNetRetrieval",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://huggingface.co/datasets/code_search_net/viewer",
        dataset={
            "path": "code_search_net",
            "revision": "fdc6a9e39575768c27eb8a2a5f702bf846eb4759",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=["Article retrieval"],
        license=None,
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="@article{husain2019codesearchnet, title={{CodeSearchNet} challenge: Evaluating the state of semantic code search}, author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc}, journal={arXiv preprint arXiv:1909.09436}, year={2019} }",
        n_samples={
            _EVAL_SPLIT: 100529,
        },
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            trust_remote_code=True,
            ** self.metadata_dict["dataset"],
        )

        self.queries = {
            self._EVAL_SPLIT: {
                str(i): row["func_documentation_string"] for i, row in enumerate(data)
            }
        }
        self.corpus = {
            self._EVAL_SPLIT: {
                str(row["func_code_url"]): {"text": row["func_code_string"]}
                for row in data
            }
        }
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                str(i): {row["func_code_url"]: 1} for i, row in enumerate(data)
            }
        }

        self.data_loaded = True
