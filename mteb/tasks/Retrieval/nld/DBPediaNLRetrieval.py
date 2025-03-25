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
        date=("2017-01-01", "2017-01-01"),  # best guess: based on publication date
        domains=["Written", "Encyclopaedic"],
        task_subtypes=[],
        license="cc-by-4.0",
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
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        adapted_from=["DBPedia"],
    )
