from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQAPLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FiQA-PL",
        description="Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        hf_hub_name="clarin-knext/fiqa-pl",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ndcg_at_10",
        revision="2e535829717f8bf9dc829b7f911cc5bbd4e6608e",
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
        n_samples={},
        avg_character_length={},
    )
