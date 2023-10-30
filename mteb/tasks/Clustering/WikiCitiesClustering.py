from ...abstasks.AbsTaskClustering import AbsTaskClustering


class WikiCitiesClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "WikiCitiesClustering",
            "hf_hub_name": "jinaai/cities_wiki_clustering",
            "description": (
                "Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia."
                "Test set includes cities from 133 countries."
            ),
            "reference": "https://huggingface.co/datasets/wikipedia",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "2af05f538fd902351ce5ec7ac84f23f71d52bae1",
        }
