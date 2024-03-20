from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class Ocnli(AbsTaskPairClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "Ocnli",
            "hf_hub_name": "C-MTEB/OCNLI",
            "description": "Original Chinese Natural Language Inference dataset",
            "reference": "https://arxiv.org/abs/2010.05444",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "ap",
            "revision": "66e76a618a34d6d565d5538088562851e6daa7ec",
        }


class Cmnli(AbsTaskPairClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "Cmnli",
            "hf_hub_name": "C-MTEB/CMNLI",
            "description": "Chinese Multi-Genre NLI",
            "reference": "https://huggingface.co/datasets/clue/viewer/cmnli",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "ap",
            "revision": "41bc36f332156f7adc9e38f53777c959b2ae9766",
        }
