from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _split_by_first_newline(s):
    # Split the string by the first newline
    parts = s.split("\n", 1)
    # Return parts or (s, '') if no newline
    return parts if len(parts) > 1 else (s, "")


class CodeRAGProgrammingSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGProgrammingSolutions",
        description="Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.",
        reference="https://arxiv.org/pdf/2406.14497",
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        dataset={
            "path": "code-rag-bench/programming-solutions",
            "revision": "1064f7bba54d5400d4836f5831fe4c2332a566a6",
        },
        date=("2024-06-02", "2024-06-02"),  # best guess
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
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
            query, doc = _split_by_first_newline(text)

            id = mt["task_id"]

            query_id = id
            doc_id = f"doc_{id}"
            self.queries[split][query_id] = query
            self.corpus[split][doc_id] = {"title": "", "text": doc}

            self.relevant_docs[split][query_id] = {
                doc_id: 1
            }  # only one correct matches
