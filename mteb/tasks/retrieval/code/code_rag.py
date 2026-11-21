from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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
    @inproceedings{wang2024coderagbenchretrievalaugmentcode,
  address = {Albuquerque, New Mexico},
  author = {Wang, Zora Zhiruo and Asai, Akari and Yu, Xinyan Velocity and Xu, Frank F. and Xie, Yiqing and Neubig, Graham and Fried, Daniel},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {NAACL} 2025},
  doi = {10.18653/v1/2025.findings-naacl.176},
  editor = {Chiruzzo, Luis and Ritter, Alan and Wang, Lu},
  isbn = {979-8-89176-195-7},
  month = {apr},
  pages = {3199--3214},
  publisher = {Association for Computational Linguistics},
  title = {{C}ode{RAG}-Bench: Can Retrieval Augment Code Generation?},
  url = {https://aclanthology.org/2025.findings-naacl.176/},
  year = {2025},
}
    """,
}


class CodeRAGProgrammingSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGProgrammingSolutions",
        description="Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.",
        dataset={
            "path": "mteb/CodeRAGProgrammingSolutions",
            "revision": "c03ca3b5de3d8d9ccd0c9e5c45c63b3b10d8f7f3",
        },
        **common_args,
    )


class CodeRAGOnlineTutorialsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGOnlineTutorials",
        description="Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.",
        dataset={
            "path": "mteb/CodeRAGOnlineTutorials",
            "revision": "39eea899325953de062731c3dd0e924d06a831bc",
        },
        **common_args,
    )


class CodeRAGLibraryDocumentationSolutionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGLibraryDocumentationSolutions",
        description="Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.",
        dataset={
            "path": "mteb/CodeRAGLibraryDocumentationSolutions",
            "revision": "0e988363415924f3b841169da53dca985534d864",
        },
        **common_args,
    )


class CodeRAGStackoverflowPostsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGStackoverflowPosts",
        description="Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.",
        dataset={
            "path": "mteb/CodeRAGStackoverflowPosts",
            "revision": "96e41c476939ff9cccfa5a493ccd664d67a048a9",
        },
        **common_args,
    )
