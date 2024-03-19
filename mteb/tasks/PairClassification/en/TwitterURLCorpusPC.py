from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterURLCorpusPC(AbsTaskPairClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "mteb/twitterurlcorpus-pairclassification",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ap",
            "revision": "8b6510b0b1fa4e4c4f879467980e9be563ec1cdf",
        }
