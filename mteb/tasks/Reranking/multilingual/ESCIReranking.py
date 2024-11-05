from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)

_EVAL_SPLIT = "test"
_LANGUAGES = {
    "us": ["eng-Latn"],
    "es": ["spa-Latn"],
    "jp": ["jpn-Jpan"],
}

_CITATION = """@article{reddy2022shopping,
    title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
    author={Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
    year={2022},
    eprint={2206.06588},
    archivePrefix={arXiv}
}"""


class ESCIReranking(AbsTaskReranking, MultilingualTask):
    metadata = TaskMetadata(
        name="ESCIReranking",
        description="",
        reference="https://github.com/amazon-science/esci-data/",
        dataset={
            "path": "mteb/esci",
            "revision": "237f74be0503482b4e8bc1b83778c7a87ea93fd8",
        },
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="map_at_1000",
        date=("2022-06-14", "2022-06-14"),
        domains=["Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
        descriptive_stats={
            "test": {
                "average_document_length": 793.9222570025365,
                "average_query_length": 20.194805194805195,
                "num_documents": 148232,
                "num_queries": 10395,
                "average_relevant_docs_per_query": 10.277825877825878,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 14.25993265993266,
                "hf_subset_descriptive_stats": {
                    "us": {
                        "average_document_length": 858.2693745556295,
                        "average_query_length": 22.554526441589484,
                        "num_documents": 87202,
                        "num_queries": 6694,
                        "average_relevant_docs_per_query": 9.446519270988945,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 13.026889752016732,
                    },
                    "es": {
                        "average_document_length": 1006.1636500281832,
                        "average_query_length": 21.262560777957862,
                        "num_documents": 31934,
                        "num_queries": 1851,
                        "average_relevant_docs_per_query": 12.038357644516477,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 17.252296056185845,
                    },
                    "jp": {
                        "average_document_length": 368.12785262579047,
                        "average_query_length": 10.588108108108107,
                        "num_documents": 29096,
                        "num_queries": 1850,
                        "average_relevant_docs_per_query": 11.524324324324324,
                        "average_instruction_length": 0,
                        "num_instructions": 0,
                        "average_top_ranked_per_query": 15.727567567567567,
                    },
                },
            }
        },
    )
