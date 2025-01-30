from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPediaNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-NL",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. DBPedia-NL is a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity",
        dataset={
            "path": "clips/beir-nl-dbpedia-entity",
            "revision": "e9c354ce0dfabd13e8808a052d0da2ace95cbef6",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2024-10-01", "2024-10-01"),
        domains=["Written", "Encyclopaedic"],
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
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
    )
