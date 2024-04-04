from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GerDaLIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GerDaLIR",
        description="The dataset consists of documents, passages and relevance labels in German.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "mteb/GerDaLIR",
            "revision": "48327de6ee192e9610f3069789719788957c7abd",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["de"],
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
    