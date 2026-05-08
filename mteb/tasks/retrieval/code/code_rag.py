import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def split_by_first_newline(s):
    # Split the string by the first newline
    parts = s.split("\n", 1)
    # Return parts or (s, '') if no newline
    return parts if len(parts) > 1 else (s, "")


common_args = {
    "reference": "https://arxiv.org/pdf/2406.14497",
    "type": "Reranking",
    "category": "t2t",
    "modalities": ["text"],
    "eval_splits": ["train"],
    "eval_langs": ["python-Code"],
    "main_score": "ndcg_at_10",
    "date": ("2024-06-02", "2024-06-02"),  # best guess
    "domains": ["Programming"],
    "task_subtypes": ["Code retrieval"],
    "license": "cc-by-sa-4.0",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": """
    @misc{wang2024coderagbenchretrievalaugmentcode,
  archiveprefix = {arXiv},
  author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
  eprint = {2406.14497},
  primaryclass = {cs.SE},
  title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
  url = {https://arxiv.org/abs/2406.14497},
  year = {2024},
}
    """,
}


class CodeRAGProgrammingSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGProgrammingSolutions",
        description="Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.",
        dataset={
            "path": "code-rag-bench/programming-solutions",
            "revision": "1064f7bba54d5400d4836f5831fe4c2332a566a6",
        },
        **common_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = hf_dataset[split]
        ds = ds.shuffle(seed=42)

        queries_dict = {}
        corpus_dict = {}
        relevant_docs = {}

        texts = ds["text"]
        meta = ds["meta"]
        for text, mt in zip(texts, meta):
            # in code-rag-bench,
            # text = query + "\n" + doc(code)
            query, doc = split_by_first_newline(text)

            id = mt["task_id"]

            query_id = id
            doc_id = f"doc_{id}"
            queries_dict[query_id] = query
            corpus_dict[doc_id] = {"title": "", "text": doc}

            relevant_docs[query_id] = {doc_id: 1}  # only one correct matches

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }


class CodeRAGOnlineTutorialsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGOnlineTutorials",
        description="Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.",
        dataset={
            "path": "code-rag-bench/online-tutorials",
            "revision": "095bb77130082e4690d6c3a031997b03487bf6e2",
        },
        **common_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = hf_dataset[split]
        ds = ds.shuffle(seed=42)

        queries_dict = {}
        corpus_dict = {}
        relevant_docs = {}

        titles = ds["title"]
        texts = ds["text"]
        parsed = ds["parsed"]
        idx = 0
        for title, text, mt in zip(titles, texts, parsed):
            # in code-rag-bench,
            # query=doc(code)
            # text=query+doc(code)
            query, doc = title, text

            query_id = str(idx)
            doc_id = f"doc_{idx}"
            queries_dict[query_id] = query
            corpus_dict[doc_id] = {"title": "", "text": doc}

            relevant_docs[query_id] = {doc_id: 1}  # only one correct matches

            idx += 1

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }


class CodeRAGLibraryDocumentationSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGLibraryDocumentationSolutions",
        description="Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.",
        dataset={
            "path": "code-rag-bench/library-documentation",
            "revision": "b530d3b5a25087d2074e731b76232db85b9e9107",
        },
        **common_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = hf_dataset[split]
        ds = ds.shuffle(seed=42)

        queries_dict = {}
        corpus_dict = {}
        relevant_docs = {}

        texts = ds["doc_content"]

        idx = 0
        for text in texts:
            # text format "document title \n document content"
            query, doc = split_by_first_newline(text)

            # some library documents doesn't have query-doc pair
            if not doc:
                continue
            query_id = str(idx)
            doc_id = f"doc_{idx}"
            queries_dict[query_id] = query
            corpus_dict[doc_id] = {"title": "", "text": doc}
            # only one correct match
            relevant_docs[query_id] = {doc_id: 1}
            idx += 1

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }


class CodeRAGStackoverflowPostsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGStackoverflowPosts",
        description="Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.",
        dataset={
            "path": "code-rag-bench/stackoverflow-posts",
            "revision": "04e05d86cb0ac467b29a5d87f4c56eac99dfc0a4",
        },
        **common_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = hf_dataset[split]
        ds = ds.shuffle(seed=42)

        queries_dict = {}
        corpus_dict = {}
        relevant_docs = {}

        texts = ds["text"]
        idx = 0
        for text in texts:
            # in code-rag-bench,
            # text = query + "\n" + doc
            query, doc = split_by_first_newline(text)

            query_id = str(idx)
            doc_id = f"doc_{idx}"
            queries_dict[query_id] = query
            corpus_dict[doc_id] = {"title": "", "text": doc}

            relevant_docs[query_id] = {doc_id: 1}  # only one correct matches
            idx += 1

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }
