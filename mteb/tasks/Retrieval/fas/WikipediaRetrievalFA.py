from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalFA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalFA",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-fa",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-fa",
            "revision": "3106996981d50c828e130f3a977bd4c5e6aea3b5",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="LM-generated and verified",
        bibtex_citation="",
        n_samples={"test": 1500},
        avg_character_length={"test": 452},
    )
