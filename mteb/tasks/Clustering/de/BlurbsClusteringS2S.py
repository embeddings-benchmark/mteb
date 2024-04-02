from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class BlurbsClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BlurbsClusteringS2S",
        description="Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        hf_hub_name="slvnwhrl/blurbs-clustering-s2s",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["de"],
        main_score="v_measure",
        revision="22793b6a6465bf00120ad525e38c51210858132c",
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
        n_samples={"test": 174637},
        avg_character_length={"test": 23.02},
    )
