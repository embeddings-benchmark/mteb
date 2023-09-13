from ...abstasks.AbsTaskClustering import AbsTaskClustering


class BigPatentClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "BigPatentClustering",
            "hf_hub_name": "jinaai/big-patent-clustering",
            "description": (
                "Clustering of documents from the Big Patent dataset. Test set only includes documents"
                "belonging to a single category, with a total of 9 categories."
            ),
            "reference": "https://huggingface.co/datasets/big_patent",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "62d5330920bca426ce9d3c76ea914f15fc83e891",
        }
