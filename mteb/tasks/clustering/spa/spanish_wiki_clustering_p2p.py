from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class SpanishWikiClusteringP2P(AbsTaskClustering):
    """Cluster Spanish Wikipedia passages from six audited thematic domains."""

    max_document_to_embed = None
    max_fraction_of_documents_to_embed = None
    max_documents_per_cluster = 1200

    metadata = TaskMetadata(
        name="SpanishWikiClusteringP2P",
        description=(
            "Thematic clustering of Spanish Wikipedia passages across six balanced, audited domains."
        ),
        reference="https://huggingface.co/datasets/ClementeH/SpanishWikiClustering-v2",
        dataset={
            "path": "mteb/SpanishWikiClustering",
            "revision": "8913161772d459ea5012368440a59bf0ea059d74",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="v_measure",
        date=("2001-01-01", "2026-08-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""
        prompt="Agrupa los pasajes enciclopédicos en español según el asunto tratado.",
    )
