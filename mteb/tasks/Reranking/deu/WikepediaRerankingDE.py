from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WikipediaRerankingDE(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WikipediaRerankingDE",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-de",
        hf_hub_name="ellamind/wikipedia-2023-11-reranking-de",
        dataset={
            "path": "ellamind/wikipedia-2023-11-reranking-de",
            "revision": "b3389bfe01ced210ced15741665e49b6ba6aec75",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
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
