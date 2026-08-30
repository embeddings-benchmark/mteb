from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class SpanishWikinewsClusteringS2S(AbsTaskClustering):
    """Cluster Spanish Wikinews articles from their titles."""

    max_document_to_embed = None
    max_fraction_of_documents_to_embed = None
    max_documents_per_cluster = 800

    metadata = TaskMetadata(
        name="SpanishWikinewsClusteringS2S",
        description=(
            "Thematic clustering of Spanish Wikinews article titles across eight news categories. "
            "The evaluation set contains 800 balanced, deduplicated articles."
        ),
        reference="https://huggingface.co/datasets/ClementeH/SpanishWikinewsClustering",
        dataset={
            "path": "ClementeH/SpanishWikinewsClustering",
            "name": "s2s",
            "revision": "494de667cecfc3f8d2fc5db7a9200bc6e56921ee",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="v_measure",
        date=("2005-01-01", "2026-08-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://creativecommons.org/licenses/by/2.5/",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{henriquez2026spanishwikinewsclustering,
  author = {Henríquez, Clemente},
  publisher = {Hugging Face},
  title = {SpanishWikinewsClustering},
  url = {https://huggingface.co/datasets/ClementeH/SpanishWikinewsClustering},
  year = {2026},
}
""",
        prompt="Identifica el tema principal de la noticia a partir del título.",
    )
