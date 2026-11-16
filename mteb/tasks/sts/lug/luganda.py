import mteb
from mteb.abstasks import AbsTaskSTS


class LugandaSTS(AbsTaskSTS):
    """Semantic textual similarity evaluation for Luganda sentence pairs."""

    metadata = mteb.TaskMetadata(
        name="LugandaSTS",
        description=(
            "Luganda Semantic Textual Similarity benchmark adapted from the "
            "STS Benchmark. It evaluates sentence embedding models by measuring "
            "the correlation between cosine similarity scores and the adapted "
            "semantic similarity scores for pairs of Luganda sentences. The "
            "current evaluation split contains 300 sentence pairs."
        ),
        dataset={
            "path": "sagepond/sts",
            "revision": "0d52f8af50b78a35d7134e6da422d3d04ca70606",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["lug-Latn"],
        main_score="spearman",
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="human-translated and localized",
        adapted_from=["STSBenchmark"],
        contributed_by="Caleb Matovu",
        is_public=True,
    )

    column_names = ("sentence1", "sentence2")
    min_score = 0
    max_score = 5