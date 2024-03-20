from ....abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "ArxivClusteringS2S",
            "hf_hub_name": "mteb/arxiv-clustering-s2s",
            "description": (
                "Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category"
            ),
            "reference": "https://www.kaggle.com/Cornell-University/arxiv",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "f910caf1a6075f7329cdf8c1a6135696f37dbd53",
        }
