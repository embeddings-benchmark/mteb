from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCS(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS",
        dataset={
            "path": "mteb/scidocs",
            "revision": "f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88",
        },
        description=(
            "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
            " prediction, to document classification and recommendation."
        ),
        reference="https://allenai.org/data/scidocs",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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
