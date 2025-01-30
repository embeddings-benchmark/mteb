from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-NL",
        dataset={
            "path": "clips/beir-nl-nq",
            "revision": "caaaf6d6e44d92733775172256472abb3a438b42",
        },
        description="NQ-NL is a translation of NQ",
        reference="https://huggingface.co/datasets/clips/beir-nl-nq",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2024-10-01", "2024-10-01"),
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""@misc{banar2024beirnlzeroshotinformationretrieval,
    title={BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language}, 
     author={Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
     year={2024},
     eprint={2412.08329},
     archivePrefix={arXiv},
     primaryClass={cs.CL},
     url={https://arxiv.org/abs/2412.08329}, 
}""",
    )
