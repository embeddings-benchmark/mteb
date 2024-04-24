from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalQuAD(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalQuAD",
        description="The dataset consists of questions and legal documents in German.",
        reference="https://github.com/Christoph911/AIKE2021_Appendix",
        dataset={
            "path": "mteb/LegalQuAD",
            "revision": "37aa6cfb01d48960b0f8e3f17d6e3d99bf1ebc3e",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Question answering"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="",
        n_samples=None,
        avg_character_length=None,
    )
