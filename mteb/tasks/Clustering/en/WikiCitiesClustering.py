from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikiCitiesClustering(AbsTaskClustering):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "WikiCitiesClustering",
            "hf_hub_name": "jinaai/cities_wiki_clustering",
            "description": (
                "Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia."
                "Test set includes 126 countries, and a total of 3531 cities."
            ),
            "reference": "https://huggingface.co/datasets/wikipedia",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "ddc9ee9242fa65332597f70e967ecc38b9d734fa",
        }
