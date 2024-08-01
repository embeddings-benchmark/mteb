from __future__ import annotations

import datasets

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"


def _load_code_search_code_retrieval(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        corpus_identifier = f"-{lang}-queries-corpus"

        dataset = datasets.load_dataset(
            path + corpus_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        ).filter(lambda example: example["partition"] == "test")

        corpus[lang][split] = {}
        corpus_data = dataset["corpus"]
        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}

        queries[lang][split] = {}
        queries_data = dataset["queries"]
        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        # Load relevant documents data
        qrels_identifier = f"-{lang}-qrels"

        qrels_data = datasets.load_dataset(
            path + qrels_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = row["query_id"]
            doc_id = row["corpus_id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class CodeSearchNetCCRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeSearchNetCCRetrieval",
        description="The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.",
        reference="https://arxiv.org/abs/2407.02883",
        dataset={
            "path": "CoIR-Retrieval/CodeSearchNet-ccr",
            "revision": "main",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2024coircomprehensivebenchmarkcode,
        title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
        author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
        year={2024},
        eprint={2407.02883},
        archivePrefix={arXiv},
        primaryClass={cs.IR},
        url={https://arxiv.org/abs/2407.02883},
        }""",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000,
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = (
            _load_code_search_code_retrieval(
                path=self.metadata_dict["dataset"]["path"],
                langs=self.hf_subsets,
                splits=self.metadata_dict["eval_splits"],
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )
        )

        self.data_loaded = True
