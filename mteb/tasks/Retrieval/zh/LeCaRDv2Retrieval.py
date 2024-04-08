from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LeCaRDv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LeCaRDv2",
        description="The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.",
        reference="https://github.com/THUIR/LeCaRDv2",
        dataset={
            "path": "mteb/LeCaRDv2",
            "revision": "b78e18688c3d012a33dc3676597c1d1b2243ce1c",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="MIT license",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
