from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFactPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-PL",
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        hf_hub_name="clarin-knext/scifact-pl",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ndcg_at_10",
        revision="47932a35f045ef8ed01ba82bf9ff67f6e109207e",
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
    )

