from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalBenchCorporateLobbying(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalBenchCorporateLobbying",
        description="The dataset includes bill titles and bill summaries related to corporate lobbying.",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying",
        dataset={
            "path": "mteb/legalbench_corporate_lobbying",
            "revision": "f69691c650464e62546d7f2a4536f8f87c891e38",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation= None,
        n_samples=None,
        avg_character_length=None,
    )
