from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalBenchCorporateLobbying(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalBenchCorporateLobbying",
        description="",
        reference="https://huggingface.co/datasets/mteb/legalbench_corporate_lobbying",
        hf_hub_name="mteb/legalbench_corporate_lobbying",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        revision="",
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
