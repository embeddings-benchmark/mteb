from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


def _load_cos_qa_retrieval(path: str, cache_dir: str = None, revision: str = None):
    split = _EVAL_SPLIT

    corpus = {split: {}}
    queries = {split: {}}
    relevant_docs = {split: {}}

    corpus_identifier = "-queries-corpus"
    dataset = datasets.load_dataset(
        path + corpus_identifier,
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    ).filter(lambda example: example["partition"] == "test")

    corpus_data = dataset["corpus"]
    for row in corpus_data:
        doc_id = row["_id"]
        doc_title = row["title"]
        doc_text = row["text"]
        corpus[split][doc_id] = {"title": doc_title, "text": doc_text}

    # Load queries data
    queries_data = dataset["queries"]

    for row in queries_data:
        query_id = row["_id"]
        query_text = row["text"]
        queries[split][query_id] = query_text

    # Load relevant documents data
    qrels_identifier = "-qrels"
    qrels_data = datasets.load_dataset(
        path + qrels_identifier,
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )

    for row in qrels_data[split]:
        query_id = row["query_id"]
        doc_id = row["corpus_id"]
        score = row["score"]
        if query_id not in relevant_docs[split]:
            relevant_docs[split][query_id] = {}
        relevant_docs[split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class CosQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CosQA",
        description="The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://arxiv.org/abs/2105.13239",
        dataset={
            "path": "CoIR-Retrieval/cosqa",
            "revision": "main",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-05-07", "2021-05-07"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{huang2021cosqa20000webqueries,
              title={CoSQA: 20,000+ Web Queries for Code Search and Question Answering}, 
              author={Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
              year={2021},
              eprint={2105.13239},
              archivePrefix={arXiv},
              primaryClass={cs.CL},
              url={https://arxiv.org/abs/2105.13239}, 
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

        self.corpus, self.queries, self.relevant_docs = _load_cos_qa_retrieval(
            path=self.metadata_dict["dataset"]["path"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
