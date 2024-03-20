from mteb.abstasks.AbsTaskReranking import AbsTaskReranking


class T2Reranking(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "T2Reranking",
            "hf_hub_name": "C-MTEB/T2Reranking",
            "description": "T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
            "reference": "https://arxiv.org/abs/2304.03679",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "map",
            "revision": "76631901a18387f85eaa53e5450019b87ad58ef9",
        }


class MMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MMarcoReranking",
            "hf_hub_name": "C-MTEB/Mmarco-reranking",
            "description": "mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
            "reference": "https://github.com/unicamp-dl/mMARCO",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "map",
            "revision": "8e0c766dbe9e16e1d221116a3f36795fbade07f6",
        }


class CMedQAv1(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CMedQAv1",
            "hf_hub_name": "C-MTEB/CMedQAv1-reranking",
            "description": "Chinese community medical question answering",
            "reference": "https://github.com/zhangsheng93/cMedQA",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "map",
            "revision": "8d7f1e942507dac42dc58017c1a001c3717da7df",
        }


class CMedQAv2(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CMedQAv2",
            "hf_hub_name": "C-MTEB/CMedQAv2-reranking",
            "description": "Chinese community medical question answering",
            "reference": "https://github.com/zhangsheng93/cMedQA2",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "map",
            "revision": "23d186750531a14a0357ca22cd92d712fd512ea0",
        }
