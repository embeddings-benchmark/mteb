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
        descriptive_stats={
            "n_samples": {"dev": 44608},
            "dev": {
                "average_document_length": 611.2265860323572,
                "average_query_length": 36.47500798466943,
                "num_documents": 128812,
                "num_queries": 12524,
                "average_relevant_docs_per_query": 2.3573139572021717,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 10.285212392206963,
                "hf_subset_descriptive_stats": {
                    "ar": {
                        "average_document_length": 693.8530670959345,
                        "average_query_length": 29.480662983425415,
                        "num_documents": 29197,
                        "num_queries": 2896,
                        "average_relevant_docs_per_query": 1.953729281767956,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.081837016574585,
                    },
                    "bn": {
                        "average_document_length": 711.114122681883,
                        "average_query_length": 46.98053527980535,
                        "num_documents": 4206,
                        "num_queries": 411,
                        "average_relevant_docs_per_query": 2.099756690997567,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.233576642335766,
                    },
                    "de": {
                        "average_document_length": 634.067007019783,
                        "average_query_length": 46.06578947368421,
                        "num_documents": 3134,
                        "num_queries": 304,
                        "average_relevant_docs_per_query": 2.6348684210526314,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.30921052631579,
                    },
                    "en": {
                        "average_document_length": 750.0277271068953,
                        "average_query_length": 40.31003811944092,
                        "num_documents": 8223,
                        "num_queries": 787,
                        "average_relevant_docs_per_query": 2.7941550190597204,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.44853875476493,
                    },
                    "es": {
                        "average_document_length": 626.9067948509044,
                        "average_query_length": 47.573743922204216,
                        "num_documents": 6137,
                        "num_queries": 617,
                        "average_relevant_docs_per_query": 4.345218800648298,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 9.946515397082658,
                    },
                    "fa": {
                        "average_document_length": 492.3620453507838,
                        "average_query_length": 41.1503164556962,
                        "num_documents": 6571,
                        "num_queries": 632,
                        "average_relevant_docs_per_query": 2.079113924050633,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.397151898734178,
                    },
                    "fi": {
                        "average_document_length": 630.257385028533,
                        "average_query_length": 38.76246830092984,
                        "num_documents": 11916,
                        "num_queries": 1183,
                        "average_relevant_docs_per_query": 1.9907016060862215,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.072696534234996,
                    },
                    "fr": {
                        "average_document_length": 558.0711577719452,
                        "average_query_length": 43.883381924198254,
                        "num_documents": 3429,
                        "num_queries": 343,
                        "average_relevant_docs_per_query": 2.131195335276968,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 9.997084548104956,
                    },
                    "hi": {
                        "average_document_length": 577.9819690898684,
                        "average_query_length": 53.34,
                        "num_documents": 3494,
                        "num_queries": 350,
                        "average_relevant_docs_per_query": 2.1485714285714286,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 9.982857142857142,
                    },
                    "id": {
                        "average_document_length": 677.5285759393748,
                        "average_query_length": 38.03407880724175,
                        "num_documents": 9501,
                        "num_queries": 939,
                        "average_relevant_docs_per_query": 3.110756123535676,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.118210862619808,
                    },
                    "ja": {
                        "average_document_length": 292.59835768626976,
                        "average_query_length": 17.7465495608532,
                        "num_documents": 8281,
                        "num_queries": 797,
                        "average_relevant_docs_per_query": 2.1543287327478042,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.39021329987453,
                    },
                    "ko": {
                        "average_document_length": 282.47890088321884,
                        "average_query_length": 21.624413145539908,
                        "num_documents": 3057,
                        "num_queries": 213,
                        "average_relevant_docs_per_query": 2.568075117370892,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 14.352112676056338,
                    },
                    "ru": {
                        "average_document_length": 767.156817659232,
                        "average_query_length": 44.15878107457899,
                        "num_documents": 13047,
                        "num_queries": 1247,
                        "average_relevant_docs_per_query": 2.8123496391339216,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.46271050521251,
                    },
                    "sw": {
                        "average_document_length": 541.4608131997643,
                        "average_query_length": 38.88565488565489,
                        "num_documents": 5091,
                        "num_queries": 481,
                        "average_relevant_docs_per_query": 1.8898128898128899,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.584199584199585,
                    },
                    "te": {
                        "average_document_length": 787.1287703016242,
                        "average_query_length": 38.464285714285715,
                        "num_documents": 862,
                        "num_queries": 84,
                        "average_relevant_docs_per_query": 1.3095238095238095,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.261904761904763,
                    },
                    "th": {
                        "average_document_length": 586.1071334214002,
                        "average_query_length": 42.83150684931507,
                        "num_documents": 7570,
                        "num_queries": 730,
                        "average_relevant_docs_per_query": 1.8356164383561644,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 10.36986301369863,
                    },
                    "yo": {
                        "average_document_length": 407.2617845117845,
                        "average_query_length": 37.6890756302521,
                        "num_documents": 1188,
                        "num_queries": 119,
                        "average_relevant_docs_per_query": 1.2100840336134453,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 9.983193277310924,
                    },
                    "zh": {
                        "average_document_length": 182.91760491299897,
                        "average_query_length": 10.859335038363172,
                        "num_documents": 3908,
                        "num_queries": 391,
                        "average_relevant_docs_per_query": 2.4910485933503836,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 9.994884910485933,
                    },
                },
            },
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
            if "name" in self.metadata_dict["dataset"]:
                cur_dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore
                assert (
                    hf_subset == "default"
                ), f"Only default subset is supported for {self.metadata.name} since `name` is given in the metadata."
            else:
                cur_dataset = datasets.load_dataset(
                    **self.metadata_dict["dataset"], name=hf_subset
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
