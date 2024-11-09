from __future__ import annotations

import logging
from collections import defaultdict

import tqdm

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MindSmallReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MindSmallReranking",
        description="Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        dataset={
            "path": "mteb/mind_small",
            "revision": "59042f120c80e8afa9cdbb224f67076cec0fc9a7",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_over_subqueries_map_at_1000",
        date=("2019-10-12", "2019-11-22"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve relevant news articles based on user browsing history",
        bibtex_citation="""@inproceedings{wu-etal-2020-mind, title = "{MIND}: A Large-scale Dataset for News 
        Recommendation", author = "Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi, 
        Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming", 
        editor = "Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel", booktitle = 
        "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", month = jul, 
        year = "2020", address = "Online", publisher = "Association for Computational Linguistics", 
        url = "https://aclanthology.org/2020.acl-main.331", doi = "10.18653/v1/2020.acl-main.331", 
        pages = "3597--3606", abstract = "News recommendation is an important technique for personalized news 
        service. Compared with product and movie recommendations which have been comprehensively studied, 
        the research on news recommendation is much more limited, mainly due to the lack of a high-quality benchmark 
        dataset. In this paper, we present a large-scale dataset named MIND for news recommendation. Constructed from 
        the user click logs of Microsoft News, MIND contains 1 million users and more than 160k English news 
        articles, each of which has rich textual content such as title, abstract and body. We demonstrate MIND a good 
        testbed for news recommendation through a comparative study of several state-of-the-art news recommendation 
        methods which are originally developed on different proprietary datasets. Our results show the performance of 
        news recommendation highly relies on the quality of news content understanding and user interest modeling. 
        Many natural language processing techniques such as effective text representation methods and pre-trained 
        language models can effectively improve the performance of news recommendation. The MIND dataset will be 
        available at https://msnews.github.io}.", }""",
    )

    def process_example(
        self, example: dict, split: str, query_idx: int, subquery_idx: int
    ) -> dict:  # Added subquery_idx parameter
        """Process a single example from the dataset."""
        query = example["query"]
        positive_docs = example["positive"]
        negative_docs = example["negative"]

        # Modified query_id to include subquery index
        query_id = f"{split}_query{query_idx}_{subquery_idx}"

        # Rest of the method remains the same
        example_data = {
            "query_id": query_id,
            "query": query,
            "doc_ids": [],
            "doc_texts": [],
            "relevance_scores": [],
        }

        def get_doc_hash(text: str) -> str:
            import hashlib

            return hashlib.md5(text.encode()).hexdigest()

        # Process positive documents
        for i, pos_doc in enumerate(positive_docs):
            doc_hash = get_doc_hash(pos_doc)
            if pos_doc in self.doc_text_to_id[split]:
                doc_id = self.doc_text_to_id[split][pos_doc]
            else:
                formatted_i = str(i).zfill(5)
                doc_id = f"apositive_{doc_hash}_{formatted_i}"
                self.doc_text_to_id[split][pos_doc] = doc_id

            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(pos_doc)
            example_data["relevance_scores"].append(1)

        # Process negative documents
        for i, neg_doc in enumerate(negative_docs):
            doc_hash = get_doc_hash(neg_doc)
            if neg_doc in self.doc_text_to_id[split]:
                doc_id = self.doc_text_to_id[split][neg_doc]
            else:
                formatted_i = str(i).zfill(5)
                doc_id = f"negative_{doc_hash}_{formatted_i}"
                self.doc_text_to_id[split][neg_doc] = doc_id

            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(neg_doc)
            example_data["relevance_scores"].append(0)

        return example_data

    def load_data(self, **kwargs):
        """Load and transform the dataset with efficient deduplication."""
        if self.data_loaded:
            return

        # Call parent class method
        super(AbsTaskRetrieval, self).load_data(**kwargs)

        logging.info(
            f"Transforming old format to standard format for {self.metadata.name}"
        )

        self.corpus = defaultdict(lambda: defaultdict(dict))
        self.queries = defaultdict(lambda: defaultdict(dict))
        self.relevant_docs = defaultdict(lambda: defaultdict(dict))
        self.top_ranked = defaultdict(lambda: defaultdict(list))
        self.doc_text_to_id = defaultdict(dict)

        # Process each split
        for split in self.dataset:
            if split == "train":
                continue
            logging.info(f"Processing split {split}")

            # Pre-allocate lists for batch processing
            all_queries = []
            all_positives = []
            all_negatives = []
            all_instance_indices = []
            all_subquery_indices = []

            # First pass: expand queries while maintaining relationships
            current_instance_idx = 0
            for instance in tqdm.tqdm(self.dataset[split]):
                queries = instance["query"]
                positives = instance.get("positive", [])
                negatives = instance.get("negative", [])

                # For each query in this instance
                for subquery_idx, query in enumerate(queries):
                    all_queries.append(query)
                    all_positives.append(positives)  # Same positives for each subquery
                    all_negatives.append(negatives)  # Same negatives for each subquery
                    all_instance_indices.append(current_instance_idx)
                    all_subquery_indices.append(subquery_idx)

                current_instance_idx += 1

            # Filter valid examples
            valid_examples = []
            valid_instance_indices = []
            valid_subquery_indices = []

            # Filter while maintaining relationships
            for idx, (pos, neg) in enumerate(zip(all_positives, all_negatives)):
                if len(pos) > 0 and len(neg) > 0:
                    valid_examples.append(idx)
                    valid_instance_indices.append(all_instance_indices[idx])
                    valid_subquery_indices.append(all_subquery_indices[idx])

            total_instances = len(set(all_instance_indices))
            valid_unique_instances = len(set(valid_instance_indices))
            logging.info(
                f"Found {total_instances} total instances, {valid_unique_instances} valid instances"
            )
            logging.info(
                f"Filtered {len(all_queries) - len(valid_examples)} invalid examples. {len(valid_examples)} remaining."
            )

            # Process valid examples in batches
            batch_size = 1000
            for batch_start in tqdm.tqdm(range(0, len(valid_examples), batch_size)):
                batch_end = min(batch_start + batch_size, len(valid_examples))
                batch_indices = valid_examples[batch_start:batch_end]

                # Process batch
                for i, example_idx in enumerate(batch_indices):
                    instance_idx = valid_instance_indices[batch_start + i]
                    subquery_idx = valid_subquery_indices[batch_start + i]

                    example_data = self.process_example(
                        {
                            "query": all_queries[example_idx],
                            "positive": all_positives[example_idx],
                            "negative": all_negatives[example_idx],
                        },
                        split,
                        instance_idx,
                        subquery_idx,
                    )

                    # Populate data structures
                    query_id = example_data["query_id"]
                    self.queries[split][query_id] = example_data["query"]

                    for doc_id, doc_text, relevance in zip(
                        example_data["doc_ids"],
                        example_data["doc_texts"],
                        example_data["relevance_scores"],
                    ):
                        if doc_id not in self.corpus[split]:
                            self.corpus[split][doc_id] = {
                                "text": doc_text,
                                "_id": doc_id,
                            }

                        self.top_ranked[split][query_id].append(doc_id)
                        self.relevant_docs[split][query_id][doc_id] = relevance

        self.instructions = None
        self.data_loaded = True
