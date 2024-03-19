from ....abstasks.AbsTaskClustering import AbsTaskClustering


class StackExchangeClusteringP2P(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "StackExchangeClusteringP2P",
            "hf_hub_name": "mteb/stackexchange-clustering-p2p",
            "description": (
                "Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k"
                " paragraphs."
            ),
            "reference": "https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        }
