from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WikipediaRerankingBN(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WikipediaRerankingBN",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-bn",
        hf_hub_name="ellamind/wikipedia-2023-11-reranking-bn",
        dataset={
            "path": "ellamind/wikipedia-2023-11-reranking-bn",
            "revision": "8f7f11b6fdb58296df57db1f0935c9697be88ef4",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ben-Beng"],
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
        n_samples=None,
        avg_character_length=None,
    )
