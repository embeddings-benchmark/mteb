from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterURLCorpusBC(AbsTaskPairClassification):
    @property
    def description(self):
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
        }
