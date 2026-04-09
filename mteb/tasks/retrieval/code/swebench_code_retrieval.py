from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEbenchCodeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEbenchCodeRetrieval",
        description="A code retrieval task based on SWE-bench Verified, a curated set of 500 real GitHub issues from 12 popular Python repositories. Each query is a GitHub issue description (bug report or feature request), and the corpus contains Python source files from the associated repositories. The task is to retrieve the source files that need to be modified to resolve each issue. This represents a realistic software engineering retrieval scenario where developers search codebases to locate relevant files for bug fixes.",
        reference="https://www.swebench.com/",
        dataset={
            "path": "embedding-benchmark/SWEbenchCodeRetrieval",
            "revision": "d875afcc6a7296022f25a59c0a671c990233d016",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2023-10-10", "2024-12-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a GitHub issue, retrieve the source code files that need to be modified to resolve the issue."
        },
        bibtex_citation=r"""
@misc{jimenez2024swebenchlanguagemodelsresolve,
  archiveprefix = {arXiv},
  author = {Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
  eprint = {2310.06770},
  primaryclass = {cs.CL},
  title = {SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  url = {https://arxiv.org/abs/2310.06770},
  year = {2024},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(
            self.metadata.dataset["path"],
            "corpus",
            revision=self.metadata.dataset["revision"],
        )["corpus"]
        queries_ds = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            revision=self.metadata.dataset["revision"],
        )["queries"]
        qrels_ds = load_dataset(
            self.metadata.dataset["path"],
            "default",
            revision=self.metadata.dataset["revision"],
        )["test"]

        # Initialize data structures with 'test' split
        corpus = {}
        queries = {}
        relevant_docs = {}

        # Process corpus
        for item in corpus_ds:
            corpus[item["id"]] = {"title": "", "text": item["text"]}

        # Process queries
        for item in queries_ds:
            queries[item["id"]] = item["text"]

        # Process qrels (relevant documents)
        for item in qrels_ds:
            query_id = item["query-id"]
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][item["corpus-id"]] = int(item["score"])

        # Organize data by splits as expected by MTEB
        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}

        self.data_loaded = True
