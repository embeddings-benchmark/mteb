from __future__ import annotations

from mteb.abstasks import AbsTaskClustering, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "da": ["dan-Latn"],
    "la": ["lav-Latn"],
    "gv": ["glv-Latn"],
    "sq": ["sqi-Latn"],
}

class WikiClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference=None,
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "3a598004635243dd5009ac5efaeb19d52384e21f",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
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
        n_samples=None,
        avg_character_length=None,
    )

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from mteb import MTEB

    # Define the sentence-transformers model name
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=[WikiClusteringP2P()])
    results = evaluation.run(model)


    # Load dataset from WikiClustering
    task = WikiClusteringP2P()
    data = task.load_data()

    import datasets
    df = datasets.load_dataset(**task.metadata_dict["dataset"])