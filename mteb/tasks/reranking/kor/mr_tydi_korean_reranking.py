from __future__ import annotations

import logging

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

_EVAL_SPLIT = "test"


def _load_mr_tydi_korean(
    path: str, splits: list[str], revision: str | None = None
) -> tuple[dict, dict, dict]:
    lang = "korean"
    corpus = {lang: {split: {} for split in splits}}
    queries = {lang: {split: {} for split in splits}}
    relevant_docs = {lang: {split: {} for split in splits}}

    split = _EVAL_SPLIT

    qrels_data = datasets.load_dataset(
        path,
        name=f"{lang}-qrels",
        revision=revision,
    )[split]
    for row in qrels_data:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = row["score"]
        if qid not in relevant_docs[lang][split]:
            relevant_docs[lang][split][qid] = {}
        relevant_docs[lang][split][qid][did] = score

    corpus_data = datasets.load_dataset(
        path,
        name=f"{lang}-corpus",
        revision=revision,
    )["train"]
    for row in corpus_data:
        corpus[lang][split][row["_id"]] = {
            "title": row["title"],
            "text": row["text"],
        }

    queries_data = datasets.load_dataset(
        path,
        name=f"{lang}-queries",
        revision=revision,
    )[split]
    for row in queries_data:
        queries[lang][split][row["_id"]] = row["text"]

    logger.info("Loaded %d Korean queries.", len(queries[lang][split]))
    return corpus, queries, relevant_docs


class MrTyDiKoreanReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTyDiKoreanReranking",
        description=(
            "Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven "
            "typologically diverse languages. It is designed for monolingual retrieval, "
            "specifically to evaluate ranking with learned dense representations. "
            "This task adapts the Korean test split for reranking evaluation."
        ),
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        dataset={
            "path": "mteb/mrtidy",
            "revision": "fc24a3ce8f09746410daee3d5cd823ff7a0675b7",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={"korean": ["kor-Kore"]},
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-08-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["MrTidyRetrieval"],
        bibtex_citation=r"""
@article{mrtydi,
  author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
  journal = {arXiv:2108.08787},
  title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
  year = {2021},
}
""",
        prompt="Given a Korean question, retrieve Wikipedia passages that answer the question",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_mr_tydi_korean(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
