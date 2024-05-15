from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalIT(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalIT",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-it",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-it",
            "revision": "0aea98043dccc5b2a4ca0306e80821bdb56f1485",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
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
