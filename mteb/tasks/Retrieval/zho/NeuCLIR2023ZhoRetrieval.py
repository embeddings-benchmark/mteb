from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2023ZhoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRZho2023",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2023-zho",
            "revision": "5cfc483211e77f82fc929d2addd441b37b6260b8",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["zho-Hans"],
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
