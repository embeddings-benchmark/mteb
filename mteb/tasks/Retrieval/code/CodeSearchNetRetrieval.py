from __future__ import annotations

import datasets

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


_LANGS = ['python', 'javascript', 'go', 'ruby', 'java', 'php']


class CodeSearchNetRetrieval(MultilingualTask, AbsTaskRetrieval):
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
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="@article{husain2019codesearchnet, title={{CodeSearchNet} challenge: Evaluating the state of semantic code search}, author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc}, journal={arXiv preprint arXiv:1909.09436}, year={2019} }",
        n_samples={
            _EVAL_SPLIT: 1000,
        },
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            trust_remote_code=True,
            streaming=True,
            **self.metadata_dict["dataset"],
        )
        data = data.shuffle(seed=42)

        # remove any leaked labels. quite common in this dataset
        data = data.map(
            lambda ex: {
                "func_code_string": ex["func_code_string"].replace(
                    ex["func_documentation_string"], ""
                )
            }
        )

        lang_subs = {lang: [] for lang in _LANGS}
        for ex in data:
            lang_subs[ex["language"]].append(ex)

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang, sub in lang_subs.items():
            sub = sub[:min(
                len(sub), self.metadata_dict["n_samples"][self._EVAL_SPLIT])]

            self.queries[lang] = {
                self._EVAL_SPLIT: {
                    str(i): row["func_documentation_string"] for i, row in enumerate(sub)
                }
            }
            self.corpus[lang] = {
                self._EVAL_SPLIT: {
                    str(row["func_code_url"]): {"text": row["func_code_string"]}
                    for row in sub
                }
            }
            self.relevant_docs[lang] = {
                self._EVAL_SPLIT: {
                    str(i): {row["func_code_url"]: 1} for i, row in enumerate(sub)
                }
            }

        self.data_loaded = True
