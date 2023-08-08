from ...abstasks.AbsTaskReranking import AbsTaskReranking


class T2Reranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2Reranking',
            'hf_hub_name': "C-MTEB/T2Reranking",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class MmarcoReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MmarcoReranking',
            'hf_hub_name': "C-MTEB/Mmarco-reranking",
            'description': 'mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
            "reference": "https://github.com/unicamp-dl/mMARCO",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv1(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv1',
            "hf_hub_name": "C-MTEB/CMedQAv1-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv2(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv2',
            "hf_hub_name": "C-MTEB/CMedQAv2-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA2",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }
