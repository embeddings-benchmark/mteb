from ...abstasks.AbsTaskBinaryClassification import AbsTaskBinaryClassification

class TwitterURLCorpusBC(AbsTaskBinaryClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "mteb/twitterurlcorpus-binaryclassification",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "BinaryClassification",
            "available_splits": ["test"],
            "main_score": "ap",
        }