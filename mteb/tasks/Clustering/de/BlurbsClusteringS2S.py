from ....abstasks.AbsTaskClustering import AbsTaskClustering


class BlurbsClusteringS2S(AbsTaskClustering):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "BlurbsClusteringS2S",
            "hf_hub_name": "slvnwhrl/blurbs-clustering-s2s",
            "description": "Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.",
            "reference": "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "9bfff9a7f8f6dc6ffc9da71c48dd48b68696471d",
        }
