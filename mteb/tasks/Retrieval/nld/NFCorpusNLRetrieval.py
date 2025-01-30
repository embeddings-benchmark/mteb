from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpusNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-NL",
        dataset={
            "path": "clips/beir-nl-nfcorpus",
            "revision": "942953e674fd0f619ff89897abb806dc3df5dd39",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is "
        "a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-nfcorpus",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2024-10-01", "2024-10-01"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=[],
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
