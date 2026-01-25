from __future__ import annotations

from collections import defaultdict

import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class ERESSReranking(AbsTaskRetrieval):
    """ERESS: E-commerce Relevance Evaluation Suite for Reranking
    
    ERESS is a comprehensive e-commerce reranking dataset designed for holistic 
    evaluation of reranking models. It includes diverse query intents including
    attribute-rich queries, navigational queries, gift/audience-specific queries,
    utility queries, and more.
    
    Dataset: https://huggingface.co/datasets/thebajajra/eress
    """

    metadata = TaskMetadata(
        name="ERESSReranking",
        description="E-commerce query-product relevance reranking dataset with graded relevance labels",
        reference="https://huggingface.co/datasets/thebajajra/eress",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",  # nDCG@5 is primary
        dataset={
            "path": "thebajajra/eress",
            "revision": "main",
        },
        date=("2026-01-23", "2026-01-23"),
        domains=["Web", "E-commerce"],
        task_subtypes=["Product Reranking", "Query-Product Relevance"],
        license="apache-2.0",
        annotations_creators="LM-generated",  # LLM ensemble annotation
        dialect=[],
        sample_creation="found",  # Real-world queries
        prompt="Rerank products by relevance to the e-commerce query",
        bibtex_citation="""
@article{Bajaj2026RexRerankers,
  title   = {{RexRerankers}: {SOTA} Rankers for Product Discovery and {AI} Assistants},
  author  = {Bajaj, Rahul and Garg, Anuj and Nupur, Jaya},
  journal = {Hugging Face Blog (Community Article)},
  year    = {2026},
  month   = jan,
  url     = {https://huggingface.co/blog/thebajajra/rexrerankers},
  urldate = {2026-01-24}
}
""",
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(self, hf_dataset=None) -> None:
        """Transform ERESS dataset format to MTEB retrieval format.
        
        ERESS format:
        - query_string: query text
        - parent_asin: product ASIN (unique product ID)
        - title: product title
        - description: product description
        - relevance_score: graded relevance score (0-1)
        
        MTEB format:
        - corpus: Dataset with id, title, text
        - queries: Dataset with id, text
        - relevant_docs: dict[query_id, dict[doc_id, score]]
        - top_ranked: dict[query_id, list[doc_id]] (for reranking)
        """
        # Initialize structures
        corpus_dict = {}  # doc_id -> {title, text, id}
        queries_dict = {}  # query_id -> {id, text}
        relevant_docs_dict = defaultdict(dict)  # query_id -> {doc_id: score}
        top_ranked_dict = defaultdict(list)  # query_id -> [doc_id, ...]
        
        # Track unique queries and products
        query_to_id = {}
        query_counter = 0
        product_counter = 0

        # Use hf_dataset if provided, otherwise use self.dataset (for backward compatibility)
        dataset_to_process = hf_dataset if hf_dataset is not None else self.dataset

        for split in self.metadata.eval_splits:
            if split not in dataset_to_process:
                continue
                
            ds = dataset_to_process[split]
            
            for row in ds:
                # Get query
                query_text = row.get("query_string", row.get("query", ""))
                if not query_text:
                    continue
                
                # Get or create query ID
                if query_text not in query_to_id:
                    query_id = f"{split}_query_{query_counter}"
                    query_to_id[query_text] = query_id
                    queries_dict[query_id] = {"id": query_id, "text": query_text}
                    query_counter += 1
                else:
                    query_id = query_to_id[query_text]
                
                # Get product information
                product_asin = row.get("parent_asin", "")
                title = row.get("title", "")
                description = row.get("description", "")
                
                # Combine title and description for document text
                doc_text = f"{title}\n{description}".strip() if description else title
                if not doc_text:
                    continue
                
                # Get or create product ID (use ASIN if available, otherwise generate)
                if product_asin:
                    doc_id = f"product_{product_asin}"
                else:
                    # Generate ID based on content hash if no ASIN
                    doc_id = f"doc_{product_counter}"
                    product_counter += 1
                
                # Store corpus entry (only once per product)
                if doc_id not in corpus_dict:
                    corpus_dict[doc_id] = {
                        "id": doc_id,
                        "title": title,
                        "text": doc_text,
                    }
                
                # Get relevance score (0-1 float from ERESS)
                score = row.get("relevance_score", 0.0)
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0
                
                # Convert float score (0-1) to integer (0-100) for pytrec_eval
                # Scale to 0-100: multiply by 100 and round to nearest integer
                # This preserves graded relevance while meeting pytrec_eval's integer requirement
                score_int = int(round(score * 100))
                
                # Store relevance (use max score if same query-doc pair appears multiple times)
                if doc_id in relevant_docs_dict[query_id]:
                    relevant_docs_dict[query_id][doc_id] = max(
                        relevant_docs_dict[query_id][doc_id], score_int
                    )
                else:
                    relevant_docs_dict[query_id][doc_id] = score_int
                
                # Add to top_ranked for reranking (all documents for each query)
                if doc_id not in top_ranked_dict[query_id]:
                    top_ranked_dict[query_id].append(doc_id)
        
        # Sort top_ranked by relevance score (descending) for each query
        for query_id in top_ranked_dict:
            top_ranked_dict[query_id].sort(
                key=lambda doc_id: relevant_docs_dict[query_id].get(doc_id, 0),
                reverse=True
            )
        
        # Convert to Dataset format and create RetrievalSplitData
        for split in self.metadata.eval_splits:
            # Filter queries and relevant_docs for this split
            split_queries = {
                qid: qdata for qid, qdata in queries_dict.items()
                if qid.startswith(f"{split}_")
            }
            split_relevant_docs = {
                qid: docs for qid, docs in relevant_docs_dict.items()
                if qid.startswith(f"{split}_")
            }
            split_top_ranked = {
                qid: docs for qid, docs in top_ranked_dict.items()
                if qid.startswith(f"{split}_")
            }
            
            # Create datasets
            corpus_dataset = Dataset.from_list(list(corpus_dict.values()))
            queries_dataset = Dataset.from_list(list(split_queries.values()))
            
            # self.dataset is already initialized in MTEB format by AbsTaskRetrieval.__init__
            self.dataset["default"][split] = RetrievalSplitData(
                corpus=corpus_dataset,
                queries=queries_dataset,
                relevant_docs=dict(split_relevant_docs),
                top_ranked=dict(split_top_ranked) if split_top_ranked else None,
            )
