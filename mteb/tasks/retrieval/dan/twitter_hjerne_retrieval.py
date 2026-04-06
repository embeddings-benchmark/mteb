from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TwitterHjerneRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TwitterHjerneRetrieval",
        dataset={
            "path": "mteb/TwitterHjerneRetrieval",
            "revision": "97ad55673cf9746f8e4b3aaa92b1bb92d82e52db",
        },
        description="Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.",
        reference="https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2006-01-01", "2024-12-31"),  # best guess
        domains=["Social", "Written"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{holm2024gllms,
  author = {Holm, Soren Vejlgaard},
  title = {Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish},
  year = {2024},
}
""",
        prompt={"query": "Retrieve answers to questions asked in Danish tweets"},
        task_subtypes=["Question answering"],
    )
