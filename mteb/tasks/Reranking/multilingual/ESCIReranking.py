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


class ESCIReranking(MultilingualTask, AbsTaskReranking):
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
        main_score="map",
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
                "num_samples": 29285,
                "num_positive": 29285,
                "num_negative": 29285,
                "avg_query_len": 19.691890046098685,
                "avg_positive_len": 9.268089465596722,
                "avg_negative_len": 1.5105002561038074,
                "hf_subset_descriptive_stats": {
                    "us": {
                        "num_samples": 21296,
                        "num_positive": 21296,
                        "num_negative": 21296,
                        "avg_query_len": 21.440833959429,
                        "avg_positive_len": 8.892515026296017,
                        "avg_negative_len": 1.1956705484598047,
                    },
                    "es": {
                        "num_samples": 3703,
                        "num_positive": 3703,
                        "num_negative": 3703,
                        "avg_query_len": 20.681609505806104,
                        "avg_positive_len": 10.561706724277613,
                        "avg_negative_len": 2.749932487172563,
                    },
                    "jp": {
                        "num_samples": 4286,
                        "num_positive": 4286,
                        "num_negative": 4286,
                        "avg_query_len": 10.146756882874476,
                        "avg_positive_len": 10.016565562295847,
                        "avg_negative_len": 2.003966402239851,
                    },
                },
            }
        },
    )
