import logging

import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_code_search_code_retrieval(  # noqa: PLR0914
    path: str, langs: list, splits: str, revision: str | None = None
):
    result = {}
    split = _EVAL_SPLIT

    for lang in langs:
        qrels_data = datasets.load_dataset(
            path,
            name=f"{lang}-qrels",
            revision=revision,
        )[split]

        relevant_docs = {}
        for row in qrels_data:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][doc_id] = score

        corpus_data = datasets.load_dataset(
            path,
            name=f"{lang}-corpus",
            revision=revision,
        )["corpus"]

        corpus_dict = {}
        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus_dict[doc_id] = {"title": doc_title, "text": doc_text}

        queries_data = datasets.load_dataset(
            path,
            name=f"{lang}-queries",
            revision=revision,
        )["queries"].filter(lambda x: x["partition"] == "test")

        queries_dict = {}
        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries_dict[query_id] = query_text

        logger.info("Loaded %d %s Queries.", len(queries_dict), split.upper())

        corpus_ds = Dataset.from_list(
            [
                {"id": k, "text": v.get("text", ""), "title": v.get("title", "")}
                for k, v in corpus_dict.items()
            ]
        )
        queries_ds = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        result[lang] = {
            split: {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
        }

    return result


class COIRCodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="COIRCodeSearchNetRetrieval",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.",
        reference="https://huggingface.co/datasets/code_search_net/",
        dataset={
            "path": "CoIR-Retrieval/CodeSearchNet",
            "revision": "4adc7bc41202b5c13543c9c886a25f340634dab3",
        },
        type="Retrieval",
        category="t2t",
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
        bibtex_citation=r"""
@article{husain2019codesearchnet,
  author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal = {arXiv preprint arXiv:1909.09436},
  title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
  year = {2019},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = _load_code_search_code_retrieval(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
