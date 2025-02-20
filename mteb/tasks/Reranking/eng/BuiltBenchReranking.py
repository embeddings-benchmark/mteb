from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class BuiltBenchReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="BuiltBenchReranking",
        description="Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.",
        reference="https://arxiv.org/abs/2411.12056",
        dataset={
            "path": "mehrzad-shahin/BuiltBench-reranking",
            "revision": "fd33b0b3454deb256be06a57e8147b32ba078ff9",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2024-06-01", "2024-11-30"),
        domains=["Engineering", "Written"],
        task_subtypes=[],
        license="cc-by-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{shahinmoghadam2024benchmarking,
    title={Benchmarking pre-trained text embedding models in aligning built asset information},
    author={Shahinmoghadam, Mehrzad and Motamedi, Ali},
    journal={arXiv preprint arXiv:2411.12056},
    year={2024}
}""",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from buit asset classification systems such as IFC and Uniclass"
        },
    )
