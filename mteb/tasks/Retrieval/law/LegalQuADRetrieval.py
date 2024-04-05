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