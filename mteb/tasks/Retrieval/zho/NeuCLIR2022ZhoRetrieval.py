from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2022ZhoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRZho2022",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022-zho",
            "revision": "48d0083e37585afecb971ebe733b59e91eb721a0",
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
