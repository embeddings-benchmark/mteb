from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def split_by_first_newline(s):
    # Split the string by the first newline
    parts = s.split("\n", 1)
    # Return parts or (s, '') if no newline
    return parts if len(parts) > 1 else (s, "")


common_args = {
    "reference": "https://arxiv.org/pdf/2406.14497",
    "type": "Reranking",
    "category": "s2s",
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
        title={CodeRAG-Bench: Can Retrieval Augment Code Generation?},
        author={Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
        year={2024},
        eprint={2406.14497},
        archivePrefix={arXiv},
        primaryClass={cs.SE},
        url={https://arxiv.org/abs/2406.14497},
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
        **common_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = self.dataset[split]  # type: ignore
        ds = ds.shuffle(seed=42)

        self.queries[split] = {}
        self.relevant_docs[split] = {}
        self.corpus[split] = {}

        texts = ds["text"]
        meta = ds["meta"]
        for text, mt in zip(texts, meta):
            # in code-rag-bench,
            # text = query + "\n" + doc(code)
            query, doc = split_by_first_newline(text)

            id = mt["task_id"]

            query_id = id
            doc_id = f"doc_{id}"
            self.queries[split][query_id] = query
            self.corpus[split][doc_id] = {"title": "", "text": doc}

            self.relevant_docs[split][query_id] = {
                doc_id: 1
            }  # only one correct matches


class CodeRAGOnlineTutorialsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGOnlineTutorials",
        description="Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.",
        dataset={
            "path": "code-rag-bench/online-tutorials",
            "revision": "095bb77130082e4690d6c3a031997b03487bf6e2",
        },
        **common_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = self.dataset[split]  # type: ignore
        ds = ds.shuffle(seed=42)

        self.queries[split] = {}
        self.relevant_docs[split] = {}
        self.corpus[split] = {}

        titles = ds["title"]
        texts = ds["text"]
        parsed = ds["parsed"]
        id = 0
        for title, text, mt in zip(titles, texts, parsed):
            # in code-rag-bench,
            # query=doc(code)
            # text=query+doc(code)
            query, doc = title, text

            query_id = str(id)
            doc_id = f"doc_{id}"
            self.queries[split][query_id] = query
            self.corpus[split][doc_id] = {"title": "", "text": doc}

            self.relevant_docs[split][query_id] = {
                doc_id: 1
            }  # only one correct matches

            id += 1


class CodeRAGLibraryDocumentationSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGLibraryDocumentationSolutions",
        description="Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.",
        dataset={
            "path": "code-rag-bench/library-documentation",
            "revision": "b530d3b5a25087d2074e731b76232db85b9e9107",
        },
        **common_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = self.dataset[split]  # type: ignore
        ds = ds.shuffle(seed=42)

        self.queries[split] = {}
        self.relevant_docs[split] = {}
        self.corpus[split] = {}

        texts = ds["doc_content"]

        id = 0
        for text in texts:
            # text format "document title \n document content"
            query, doc = split_by_first_newline(text)

            # some library documents doesn't have query-doc pair
            if not doc:
                continue
            query_id = str(id)
            doc_id = f"doc_{id}"
            self.queries[split][query_id] = query
            self.corpus[split][doc_id] = {"title": "", "text": doc}
            # only one correct match
            self.relevant_docs[split][query_id] = {doc_id: 1}
            id += 1


class CodeRAGStackoverflowPostsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGStackoverflowPosts",
        description="Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.",
        dataset={
            "path": "code-rag-bench/stackoverflow-posts",
            "revision": "04e05d86cb0ac467b29a5d87f4c56eac99dfc0a4",
        },
        **common_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        split = self.metadata.eval_splits[0]
        ds: datasets.Dataset = self.dataset[split]  # type: ignore
        ds = ds.shuffle(seed=42)

        self.queries[split] = {}
        self.relevant_docs[split] = {}
        self.corpus[split] = {}

        texts = ds["text"]
        id = 0
        for text in texts:
            # in code-rag-bench,
            # text = query + "\n" + doc
            query, doc = split_by_first_newline(text)

            query_id = str(id)
            doc_id = f"doc_{id}"
            self.queries[split][query_id] = query
            self.corpus[split][doc_id] = {"title": "", "text": doc}

            self.relevant_docs[split][query_id] = {
                doc_id: 1
            }  # only one correct matches
            id += 1
