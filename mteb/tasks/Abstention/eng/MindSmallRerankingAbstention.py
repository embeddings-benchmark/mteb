from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Reranking.eng.MindSmallReranking import MindSmallReranking


class MindSmallRerankingAbstention(AbsTaskAbstention, MindSmallReranking):
    abstention_task = "Reranking"
    metadata = TaskMetadata(
        name="MindSmallRerankingAbstention",
        description="Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        hf_hub_name="mteb/mind_small",
        dataset={
            "path": "mteb/mind_small",
            "revision": "3bdac13927fdc888b903db93b2ffdbd90b295a69",
        },
        type="Abstention",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
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
        n_samples={"test": 107968},
        avg_character_length={"test": 70.9},
    )
