from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalDE(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalDE",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-de",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-de",
            "revision": "b90d255229ec4d2fed3d65647ea0f82854f4c762",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
