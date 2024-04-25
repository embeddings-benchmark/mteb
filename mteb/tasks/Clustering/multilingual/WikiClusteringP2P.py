from __future__ import annotations

from mteb.abstasks import AbsTaskClustering, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

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


class WikiClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation=None,  # None exists
        n_samples={"test": 71680},
        avg_character_length={"test": 625.3},
    )
