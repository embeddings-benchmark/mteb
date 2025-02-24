from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
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


class ESCIReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ESCIReranking",
        description="",
        reference="https://github.com/amazon-science/esci-data/",
        dataset={
            "path": "mteb/ESCIReranking",
            "revision": "dc2cfaf4fcbf238806a02ae8607786e88112463e",
        },
        type="Reranking",
        category="t2t",
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
    )
