from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NeuCLIR2022RusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIRRus2022",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022-rus",
            "revision": "e8f8b5d60068d2e699ecf8bad8df1d30e445f98f",
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
