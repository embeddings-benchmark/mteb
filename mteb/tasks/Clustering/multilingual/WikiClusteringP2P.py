from __future__ import annotations

from mteb.abstasks import AbsTaskClustering, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "da": ["dan-Latn"],
    "lv": ["lav-Latn"],
    "gv": ["glv-Latn"],
    "sq": ["sqi-Latn"],
}

class WikiClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "eb5e669fbb0471a306602ebd9084545f56560abd",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=None,
        form=None,
        domains=["Encyclopaedic"],
        task_subtypes=None,
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators=None,
        dialect=None,
        text_creation="created",
        bibtex_citation=None,
        n_samples={"test": 60000},
        avg_character_length={"test": 572.05},
    )
