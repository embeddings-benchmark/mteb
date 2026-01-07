from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    revision: str | None = None,
):
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query_id']}",
                "text": x["query"],
                "modality": "text",
            },
            remove_columns=["query_id", "query"],
        )
        queries[split] = query_ds

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus_id']}",
                "modality": "image",
            },
            remove_columns=["corpus_id"],
        )
        corpus[split] = corpus_ds

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
        )
        relevant_docs[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query_id']}"
            did = f"corpus-{split}-{row['corpus_id']}"
            if qid not in relevant_docs[split]:
                relevant_docs[split][qid] = {}
            relevant_docs[split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class KoVidore2CybersecurityRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2CybersecurityRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Cybersecurity, is a corpus of technical reports on cyber threat trends and security incident responses in Korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-cybersecurity-beir",
            "revision": "006dcb0e8f63c9736687cb36e725769c903054b0",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""
@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  year = {2026},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains}
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class KoVidore2EconomicRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2EconomicRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Economic trends, is a corpus of periodic reports on major economic indicators in Korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-economic-beir",
            "revision": "8400656ad1e90e7662d7cda44628eaa2d29ea8d8",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""
@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  year = {2026},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains}
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class KoVidore2EnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2EnergyRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy, is a corpus of reports on energy market trends, policy planning, and industry statistics, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-energy-beir",
            "revision": "17fea125be86500c0d7891967ca0e4ada14fbe0d",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""
@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  year = {2026},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains}
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class KoVidore2HrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2HrRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports on workforce outlook and employment policy in korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-hr-beir",
            "revision": "0641db2d66968538823af3a847257ee6b813c57e",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""
@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  year = {2026},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains}
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
