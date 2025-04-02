from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCSNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-NL",
        dataset={
            "path": "clips/beir-nl-scidocs",
            "revision": "4e018aa220029f9d1bd5a31de3650e322e32ea38",
        },
        description=(
            "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
            + " prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation."
        ),
        reference="https://huggingface.co/datasets/clips/beir-nl-scidocs",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2020-05-01", "2020-05-01"),  # best guess: based on submission date
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
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
        adapted_from=["SCIDOCS"],
    )
