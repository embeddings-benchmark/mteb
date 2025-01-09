from __future__ import annotations

import logging
from collections import defaultdict

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking

logger = logging.getLogger(__name__)

_EVAL_SPLIT = "dev"
_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "yo": ["yor-Latn"],
    "zh": ["zho-Hans"],
}

_CITATION = """@article{10.1162/tacl_a_00595,
    author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1114-1131},
    year = {2023},
    month = {09},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00595},
}"""


class MIRACLReranking(AbsTaskReranking, MultilingualTask):
    metadata = TaskMetadata(
        name="MIRACLReranking",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        reference="https://project-miracl.github.io/",
        dataset={
            "path": "miracl/mmteb-miracl-reranking",
            "revision": "6d1962c527217f8927fca80f890f14f36b2802af",
            "trust_remote_code": True,
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-06-01", "2023-01-30"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
    )

    def process_example(self, example: dict, split: str, query_idx: int) -> dict:
        """Process a single example from the dataset. Slightly altered from the original class"""
        query = example["query"]
        assert isinstance(query, str)
        positive_docs = set(example["positive"])
        candidate_docs = example["candidates"]

        # add four leading zeros
        # query_id = f"{split}_query{query_idx:04d}"
        query_id = f"{split}_query{query_idx}"

        # Initialize the structures for this example
        example_data = {
            "query_id": query_id,
            "query": query,
            "doc_ids": [],
            "doc_texts": [],
            "relevance_scores": [],
        }

        for i, candidate_doc in enumerate(candidate_docs):
            # format i as a five digit number
            formatted_i = str(i).zfill(5)
            doc_id = f"candidate_{query_id}_{formatted_i}"
            example_data["doc_ids"].append(doc_id)
            example_data["doc_texts"].append(candidate_doc)
            if candidate_doc in positive_docs:
                example_data["relevance_scores"].append(1)
            else:
                # this is not technically correct, but was done in the original so keeping it
                example_data["relevance_scores"].append(0)

        return example_data

    def load_data(self, **kwargs):
        """Super method to load the data, then convert to the new format. It is almost the same as the above, except there are negatives, positives, and candidates"""
        logging.info(
            f"Transforming old format to standard format for {self.metadata.name}"
        )

        self.corpus = defaultdict(lambda: defaultdict(dict))
        self.queries = defaultdict(lambda: defaultdict(dict))
        self.relevant_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.top_ranked = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        for hf_subset in hf_subsets:
            if "name" in self.metadata.dataset:
                cur_dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
                assert (
                    hf_subset == "default"
                ), f"Only default subset is supported for {self.metadata.name} since `name` is given in the metadata."
            else:
                cur_dataset = datasets.load_dataset(
                    **self.metadata.dataset, name=hf_subset
                )  # type: ignore

            for split in cur_dataset:
                # Create an enumerated dataset to pass indices
                enumerated_dataset = datasets.Dataset.from_dict(
                    {
                        "index": range(len(cur_dataset[split])),
                        "query": cur_dataset[split]["query"],
                        "positive": cur_dataset[split]["positive"],
                        "negative": cur_dataset[split]["negative"],
                        "candidates": cur_dataset[split]["candidates"],
                    }
                )

                # first, only keep those that have positives and negatives
                enumerated_dataset = enumerated_dataset.filter(
                    lambda example: len(example["positive"]) > 0
                    and len(example["negative"]) > 0
                )

                logger.info(
                    f"Filtered out {len(cur_dataset[split]) - len(enumerated_dataset)} examples. {len(enumerated_dataset)} examples remaining."
                )

                # Map the transformation function over the dataset
                processed_dataset = enumerated_dataset.map(
                    lambda example, idx: self.process_example(example, split, idx),
                    with_indices=True,
                    remove_columns=enumerated_dataset.column_names,
                )

                # Populate the data structures
                for idx, item in enumerate(processed_dataset):
                    query_id = item["query_id"]
                    self.queries[hf_subset][split][query_id] = item["query"]

                    # Add documents and relevance information
                    for doc_id, doc_text, relevance in zip(
                        item["doc_ids"], item["doc_texts"], item["relevance_scores"]
                    ):
                        self.corpus[hf_subset][split][doc_id] = {
                            "text": doc_text,
                            "_id": doc_id,
                        }
                        self.top_ranked[hf_subset][split][query_id].append(doc_id)
                        self.relevant_docs[hf_subset][split][query_id][doc_id] = (
                            relevance
                        )

                    if len(self.top_ranked[hf_subset][split][query_id]) == 0:
                        # give it a negative, even though qrels should be empty since that was how it was done in the original
                        neg_doc = cur_dataset[split]["negative"][idx][0]
                        assert isinstance(
                            neg_doc, str
                        ), f"Negative document is not a string: {neg_doc}"
                        neg_doc_id = f"negative_{query_id}"
                        self.top_ranked[hf_subset][split][query_id].append(neg_doc_id)
                        self.corpus[hf_subset][split][neg_doc_id] = {
                            "text": neg_doc,
                            "_id": neg_doc_id,
                        }
                        assert self.relevant_docs[hf_subset][split][query_id] == {}
                        logger.warning(
                            f"Query {query_id} has no relevant documents. Adding a negative example."
                        )

        self.instructions = None
        self.data_loaded = True
