from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2023FasRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRFas2023",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023-fas",
            "revision": "5ec8a0b336cbc36ae7b7f73dd106485aebdb525a",
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
