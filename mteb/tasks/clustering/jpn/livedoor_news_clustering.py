from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class LivedoorNewsClusteringv2(AbsTaskClustering):
    max_document_to_embed = 1107
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="LivedoorNewsClustering.v2",
        description="Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics). Version 2 updated on LivedoorNewsClustering by removing pairs where one of entries contain an empty sentences.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/LivedoorNewsClustering.v2",
            "revision": "21637d8f5a8978f029d1175e7e70fd0e9d9ed51e",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2000-01-01", "2014-02-09"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nd-2.1-jp",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["LivedoorNewsClustering"],
    )


class LivedoorNewsClustering(AbsTaskClustering):
    max_document_to_embed = 1107
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="LivedoorNewsClustering",
        description="Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics).",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/LivedoorNewsClustering",
            "revision": "4e1f50751b9dc2bbcbf56e3ebde55f82b74b6988",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2000-01-01", "2014-02-09"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nd-2.1-jp",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        superseded_by="LivedoorNewsClustering.v2",
    )
