from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "bs": ["bos-Latn"],
    "ca": ["cat-Latn"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "eu": ["eus-Latn"],
    "gv": ["glv-Latn"],
    "ilo": ["ilo-Latn"],
    "ku": ["kur-Latn"],
    "lv": ["lav-Latn"],
    "min": ["min-Latn"],
    "mt": ["mlt-Latn"],
    "sco": ["sco-Latn"],
    "sq": ["sqi-Latn"],
    "wa": ["wln-Latn"],
}


class WikiClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",  # None exists
        superseded_by="WikiClusteringP2P.v2",
    )


class WikiClusteringFastP2P(AbsTaskClustering):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="WikiClusteringP2P.v2",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "mteb/WikiClusteringP2P.v2",
            "revision": "596293afadbf41fd03571343f9f3d2b869b4f2e3",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",  # None exists
        adapted_from=["WikiClusteringP2P"],
    )
