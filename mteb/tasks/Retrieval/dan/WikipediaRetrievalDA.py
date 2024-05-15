from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalDA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalDA",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-da",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-da",
            "revision": "335ee8f323e62e0df5bc619d51a5fa1de438bf3f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2023-11", "2024-05"),
        form="written",
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation=["created", "LM-generated and verified"],
        bibtex_citation=None,
        n_samples={"test": 1500},
        avg_character_length=None,
    )
