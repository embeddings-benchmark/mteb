from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"
_LANGUAGES = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "zh": ["zho-Hans"],
}


class MultiLongDocReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiLongDocReranking",
        description=(
            "Reranking version of MultiLongDocRetrieval (MLDR). MLDR is a Multilingual Long-Document "
            "Retrieval dataset built on Wikipedia, Wudao and mC4, covering 13 typologically diverse languages. "
            "Specifically, we sample lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose "
            "paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. "
            "The generated question and the sampled article constitute a new text pair to the dataset."
        ),
        reference="https://huggingface.co/datasets/Shitao/MLDR",
        dataset={
            "path": "Shitao/MLDR",
            "revision": "b38336456b0e2a0dc1f6b8b3ce3a0e1f3c436d16",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=(
            "2000-01-01",
            "2024-12-31",
        ),  # Not found in the paper, guessed using the paper's publication date and constituent datasets
        domains=[
            "Encyclopaedic",
            "Written",
            "Web",
            "Non-fiction",
            "Fiction",
        ],  # narrativeqa, wikipedia, wudao, mC4
        task_subtypes=[],
        license="mit",
        annotations_creators="LM-generated",  # gpt-3.5
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{bge-m3,
  archiveprefix = {arXiv},
  author = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
  eprint = {2402.03216},
  primaryclass = {cs.CL},
  title = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  year = {2024},
}
""",
        prompt={
            "query": "Given a question, rerank long documents based on their relevance to answer the question"
        },
        adapted_from=["MultiLongDocRetrieval"],
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace."""
        if self.data_loaded:
            return

        # Load the dataset for each language
        for lang in self.metadata.eval_langs:
            self.dataset[lang] = {}
            raw_dataset = load_dataset(
                self.metadata.dataset["path"],
                name=lang,
                revision=self.metadata.dataset["revision"],
            )

            # Transform each split immediately
            for split_name in raw_dataset.keys():
                if split_name in self.metadata.eval_splits:
                    self.dataset[lang][split_name] = self._transform_split(
                        raw_dataset[split_name]
                    )

        self.data_loaded = True

    def _transform_split(self, split_data: Dataset) -> RetrievalSplitData:
        """Transform a single split of MLDR data to RetrievalSplitData format."""
        # Build corpus, queries, relevant_docs, and top_ranked
        corpus_dict = {}
        queries_list = []
        relevant_docs = {}
        top_ranked = {}

        for sample in split_data:
            query_id = sample["query_id"]

            # Add query (avoid duplicates)
            if query_id not in relevant_docs:
                queries_list.append({"id": query_id, "text": sample["query"]})
                relevant_docs[query_id] = {}
                top_ranked[query_id] = []

            # Add positive documents
            for pos_passage in sample["positive_passages"]:
                doc_id = pos_passage["docid"]
                if doc_id not in corpus_dict:
                    corpus_dict[doc_id] = {
                        "id": doc_id,
                        "text": pos_passage.get("text", ""),
                        "title": "",
                    }
                relevant_docs[query_id][doc_id] = 1
                if doc_id not in top_ranked[query_id]:
                    top_ranked[query_id].append(doc_id)

            # Add negative documents
            for neg_passage in sample["negative_passages"]:
                doc_id = neg_passage["docid"]
                if doc_id not in corpus_dict:
                    corpus_dict[doc_id] = {
                        "id": doc_id,
                        "text": neg_passage.get("text", ""),
                        "title": "",
                    }
                # Negative docs get score 0
                if doc_id not in relevant_docs[query_id]:
                    relevant_docs[query_id][doc_id] = 0
                if doc_id not in top_ranked[query_id]:
                    top_ranked[query_id].append(doc_id)

        # Convert to Dataset objects
        corpus_dataset = Dataset.from_list(list(corpus_dict.values()))
        queries_dataset = Dataset.from_list(queries_list)

        # Create and return RetrievalSplitData
        return RetrievalSplitData(
            corpus=corpus_dataset,
            queries=queries_dataset,
            relevant_docs=relevant_docs,
            top_ranked=top_ranked,
        )
