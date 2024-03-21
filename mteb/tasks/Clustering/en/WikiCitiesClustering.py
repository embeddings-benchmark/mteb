from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikiCitiesClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiCitiesClustering",
        description="Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia.",
        reference="https://huggingface.co/datasets/wikipedia",
        hf_hub_name="mteb/wikipedia-clustering",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="ddc9ee9242fa65332597f70e967ecc38b9d734fa",
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
        n_samples={},
        avg_character_length={},
    )
