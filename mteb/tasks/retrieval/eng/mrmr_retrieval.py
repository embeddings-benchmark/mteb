from __future__ import annotations

from datasets import Value, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


def _fill_none_text(texts: list[str | None]) -> dict:
    return {"text": [t if t is not None else "" for t in texts]}


def _load_mrmr(
    task: AbsTaskRetrieval,
    num_proc: int | None = None,
    build_top_ranked: bool = False,
) -> None:
    """Shared loader for the MRMR subsets.

    The corpus mixes text-only and interleaved image-text documents, so the raw
    `text`/`image` columns contain None for the missing modality. Every corpus
    document however provides a `vision` column with a unified visual rendering
    (the image itself, or the text rendered as an image), which we use as the
    image side. None texts are replaced by empty strings so that every document
    and query exposes a complete image+text pair.
    """
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]

    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    corpus = corpus.select_columns(["id", "text", "vision"])
    corpus = corpus.rename_column("vision", "image")
    corpus = corpus.cast_column("id", Value("string"))
    corpus = corpus.map(
        _fill_none_text, input_columns=["text"], batched=True, num_proc=num_proc
    )

    queries = load_dataset(path, "query", split="test", revision=revision)
    queries = queries.select_columns(["id", "text", "image", "modality"])
    queries = queries.cast_column("id", Value("string"))
    # The knowledge subset contains 2 text-only queries (of 555); the evaluation
    # pipeline requires a uniform query schema, so image-less queries are
    # excluded. All other subsets have an image for every query.
    queries = queries.filter(
        lambda mods: ["image" in m for m in mods],
        input_columns=["modality"],
        batched=True,
        num_proc=num_proc,
    )
    queries = queries.remove_columns(["modality"])
    queries = queries.map(
        _fill_none_text, input_columns=["text"], batched=True, num_proc=num_proc
    )

    # Some subsets name the qrels columns query_id/corpus_id, others
    # query-id/corpus-id; normalize before building the mapping.
    qrels_ds = load_dataset(path, "qrels", split="test", revision=revision)
    renames = {c: c.replace("_", "-") for c in qrels_ds.column_names if "_id" in c}
    if renames:
        qrels_ds = qrels_ds.rename_columns(renames)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(
            row["score"]
        )

    queries = queries.filter(
        lambda ids: [i in qrels for i in ids],
        input_columns=["id"],
        batched=True,
        num_proc=num_proc,
    )

    top_ranked = None
    if build_top_ranked:
        # Each negation query q has exactly four candidate documents whose corpus
        # ids are the query id plus a candidate suffix (q_1..q_4), and the paper
        # scores this subset as Hit@1 over those four candidates. Expose the
        # candidate sets as a top_ranked split so the task reranks each query's
        # four candidates rather than the full corpus.
        groups: dict[str, list[str]] = {}
        for cid in corpus["id"]:
            groups.setdefault(cid.rsplit("_", 1)[0], []).append(cid)
        top_ranked = {q: sorted(groups[q]) for q in queries["id"] if q in groups}

    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=corpus,
                queries=queries,
                relevant_docs=qrels,
                top_ranked=top_ranked,
            )
        }
    }
    task.data_loaded = True


_BIBTEX = r"""
@misc{zhang2025mrmr,
  archiveprefix = {arXiv},
  author = {Siyue Zhang and Yuan Gao and Xiao Zhou and Yilun Zhao and Tingyu Song and Arman Cohan and Anh Tuan Luu and Chen Zhao},
  eprint = {2510.09510},
  primaryclass = {cs.IR},
  title = {MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval},
  url = {https://arxiv.org/abs/2510.09510},
  year = {2025},
}
"""
_REFERENCE = "https://arxiv.org/abs/2510.09510"
_DESCRIPTION = (
    "MRMR is an expert-level benchmark for reasoning-intensive multimodal retrieval. "
    "Queries and corpus documents are interleaved image-text sequences, and retrieval "
    "requires domain reasoning rather than surface matching. "
)


class MRMRNegationRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MRMRNegationRetrieval",
        description=_DESCRIPTION
        + "The negation subset requires retrieving the image that satisfies a caption "
        "with negated constraints (contradiction retrieval), e.g. an image containing "
        "certain objects but not others. Each query has exactly four candidate "
        "documents (exposed as a top_ranked split), so the task reranks the four "
        "candidates and is scored by Hit@1, matching the paper's protocol.",
        reference=_REFERENCE,
        dataset={
            "path": "MRMRbenchmark/negation",
            "revision": "ee5e669c8b257790a9c66c4d9464f6d77f33636b",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2025-01-01", "2025-10-01"),
        domains=["Scene"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given an image, retrieve descriptions that have contradictory information with the image."
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mrmr(self, num_proc=num_proc, build_top_ranked=True)


class MRMRDesignRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MRMRDesignRetrieval",
        description=_DESCRIPTION
        + "The design subset covers design documents where relevance requires reasoning "
        "over visual layouts and design intent.",
        reference=_REFERENCE,
        dataset={
            "path": "MRMRbenchmark/design",
            "revision": "fc6325cdffd332e69da9144d55a0fdb4cfb98848",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-10-01"),
        domains=["Engineering"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given a vehicle design, retrieve the design requirements that it violates."
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mrmr(self, num_proc=num_proc)


class MRMRTrafficRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MRMRTrafficRetrieval",
        description=_DESCRIPTION
        + "The traffic subset requires reasoning over traffic scenes, signs and rules "
        "to retrieve the relevant document.",
        reference=_REFERENCE,
        dataset={
            "path": "MRMRbenchmark/traffic",
            "revision": "5de2aa10c61753c2c23cbf8c42b2743859c709a5",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-10-01"),
        domains=["Scene"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given a traffic case, retrieve the driving rule documents that it violates."
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mrmr(self, num_proc=num_proc)


class MRMRKnowledgeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MRMRKnowledgeRetrieval",
        description=_DESCRIPTION
        + "The knowledge subset requires multidisciplinary world knowledge to connect "
        "queries with interleaved image-text evidence documents.",
        reference=_REFERENCE,
        dataset={
            "path": "MRMRbenchmark/knowledge",
            "revision": "c4e9395dfd01e875e5b4d204d2f9fd285c2a9d2e",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-10-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve relevant documents that help answer the question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mrmr(self, num_proc=num_proc)


class MRMRTheoremRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MRMRTheoremRetrieval",
        description=_DESCRIPTION
        + "The theorem subset requires mathematical reasoning to retrieve the document "
        "stating the theorem or result relevant to the query.",
        reference=_REFERENCE,
        dataset={
            "path": "MRMRbenchmark/theorem",
            "revision": "2a4b0d370ed0e2d322cf092ab1d35348997b3fed",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-10-01"),
        domains=["Academic"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve relevant theorems that are involved in solving the problem."
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mrmr(self, num_proc=num_proc)
