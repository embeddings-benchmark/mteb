from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from sentence_transformers import SentenceTransformer
from mteb.abstasks.TaskMetadata import TaskMetadata
import datasets
class CodeRAGOnlineTutorialsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeRAGOnlineTutorials",
        description="Ranking of related scientific papers based on their title.",
        reference="https://arxiv.org/pdf/2406.14497",
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        dataset={
            "path": "code-rag-bench/online-tutorials",
            "revision": "095bb77130082e4690d6c3a031997b03487bf6e2"
        },
        date=("2024-06-02","2024-06-02"), # best guess
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
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            split = "test"
            
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
                # doc_id
                # code_id
                # query id
                query, doc = title, text
                
                
                
                query_id = str(id)
                doc_id = f"doc_{id}"
                self.queries[split][query_id] = query
                self.corpus[split][doc_id] = {"title": "", "text": doc}

                self.relevant_docs[split][query_id] = {
                    doc_id: 1
                }  # only one correct matches
                
                id += 1
