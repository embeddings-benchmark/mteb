from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2023RusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRRus2023",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023-rus",
            "revision": "d37f24d91ab6d62c06d293b8c5ff2726e85c84ba",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_20",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
