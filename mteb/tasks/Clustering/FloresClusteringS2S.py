from ...abstasks.AbsTaskClustering import AbsTaskClustering


class FloresClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FloresClusteringS2S",
            "hf_hub_name": "jinaai/flores_clustering",
            "description": (
                "Clustering of sentences from various web articles, 32 topics in total."
            ),
            "reference": "https://huggingface.co/datasets/facebook/flores",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
            "revision": "b5d0b2bd6ba007bc9145120b47818e3d3e5d1735",
        }
