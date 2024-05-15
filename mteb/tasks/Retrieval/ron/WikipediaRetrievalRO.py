from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalRO(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalRO",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-ro",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-ro",
            "revision": "b919c6df99225e82d56090f79dca3d83c7b80d09",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
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
