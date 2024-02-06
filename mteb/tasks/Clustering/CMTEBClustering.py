from ...abstasks.AbsTaskClustering import AbsTaskClustering


class CLSClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "CLSClusteringS2S",
            "hf_hub_name": "C-MTEB/CLSClusteringS2S",
            "description": (
                "Clustering of titles from CLS dataset. Clustering of 13 sets, based on the main category."
            ),
            "reference": "https://arxiv.org/abs/2209.05034",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "v_measure",
            'revision': 'e458b3f5414b62b7f9f83499ac1f5497ae2e869f',
        }



class CLSClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "CLSClusteringP2P",
            "hf_hub_name": "C-MTEB/CLSClusteringP2P",
            "description": (
                "Clustering of titles + abstract from CLS dataset. Clustering of 13 sets, based on the main category."
            ),
            "reference": "https://arxiv.org/abs/2209.05034",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "v_measure",
            'revision': '4b6227591c6c1a73bc76b1055f3b7f3588e72476',
        }



class ThuNewsClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            'name': 'ThuNewsClusteringS2S',
            'hf_hub_name': 'C-MTEB/ThuNewsClusteringS2S',
            'description': 'Clustering of titles from the THUCNews dataset',
            "reference": "http://thuctc.thunlp.org/",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "v_measure",
            'revision': '8a8b2caeda43f39e13c4bc5bea0f8a667896e10d',
        }


class ThuNewsClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            'name': 'ThuNewsClusteringP2P',
            'hf_hub_name': 'C-MTEB/ThuNewsClusteringP2P',
            'description': 'Clustering of titles + abstracts from the THUCNews dataset',
            "reference": "http://thuctc.thunlp.org/",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "v_measure",
            'revision': '5798586b105c0434e4f0fe5e767abe619442cf93',
        }
