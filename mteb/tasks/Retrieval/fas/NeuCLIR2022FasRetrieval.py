from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2022FasRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRFas2022",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022-fas",
            "revision": "fa3b9068b9318d13ed5f065d9bbd5c8aff2dcfec",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
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
