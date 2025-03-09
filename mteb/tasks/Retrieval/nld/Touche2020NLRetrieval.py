from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Touche2020NL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020-NL",
        description="Touché Task 1: Argument Retrieval for Controversial Questions. Touche2020-NL is a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-webis-touche2020",
        dataset={
            "path": "clips/beir-nl-webis-touche2020",
            "revision": "b69e63cbe72c5ce0489f69e88c35f51f13a3f993",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation="""@misc{banar2024beirnlzeroshotinformationretrieval,
    title={BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
     author={Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
     year={2024},
     eprint={2412.08329},
     archivePrefix={arXiv},
     primaryClass={cs.CL},
     url={https://arxiv.org/abs/2412.08329},
}""",
        adapted_from=["Touche2020"],
    )
