from __future__ import annotations

import logging

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_code_search_code_retrieval(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    queries = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        qrels_data = datasets.load_dataset(
            path,
            name=f"{lang}-qrels",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )[split]

        for row in qrels_data:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

        corpus_data = datasets.load_dataset(
            path,
            name=f"{lang}-corpus",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )["corpus"]

        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}

        queries_data = datasets.load_dataset(
            path,
            name=f"{lang}-queries",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )["queries"].filter(lambda x: x["partition"] == "test")

        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        queries = queries
        logger.info("Loaded %d %s Queries.", len(queries), split.upper())

    return corpus, queries, relevant_docs


class CodeSearchNetCCRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeSearchNetCCRetrieval",
        description="The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.",
        reference="https://arxiv.org/abs/2407.02883",
        dataset={
            "path": "CoIR-Retrieval/CodeSearchNet-ccr",
            "revision": "6e1effa2c03723c5fde48ee912b5ee08d4f211e8",
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
        license="mit",
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
            "avg_character_length": {
                "test": {
                    "python": {
                        "average_document_length": 388.31577184555965,
                        "average_query_length": 551.7934039415471,
                        "num_documents": 280652,
                        "num_queries": 14918,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "javascript": {
                        "average_document_length": 276.0730050152605,
                        "average_query_length": 443.70707991491946,
                        "num_documents": 65201,
                        "num_queries": 3291,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "go": {
                        "average_document_length": 185.0307932251621,
                        "average_query_length": 233.76803742920464,
                        "num_documents": 182735,
                        "num_queries": 8122,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ruby": {
                        "average_document_length": 214.86204146730464,
                        "average_query_length": 266.8731165741475,
                        "num_documents": 27588,
                        "num_queries": 1261,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "java": {
                        "average_document_length": 281.96280259139183,
                        "average_query_length": 342.5341853035144,
                        "num_documents": 181061,
                        "num_queries": 10955,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "php": {
                        "average_document_length": 268.9752569556027,
                        "average_query_length": 336.62194947909234,
                        "num_documents": 268237,
                        "num_queries": 14014,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
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
