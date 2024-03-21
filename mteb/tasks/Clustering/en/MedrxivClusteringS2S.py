from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MedrxivClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MedrxivClusteringS2S",
        description="Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        hf_hub_name="mteb/medrxiv-clustering-s2s",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="35191c8c0dca72d8ff3efcd72aa802307d469663",
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
    )
