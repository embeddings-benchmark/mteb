from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikipediaRetrievalSR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalSR",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-sr",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-sr",
            "revision": "76c4b999346718e7412697a14e18332c02b73cea",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["srp-Cyrl"],
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=None,
        text_creation="LM-generated and verified",
        bibtex_citation=None,
        n_samples={"test": 1500},
        avg_character_length=None,
    )
