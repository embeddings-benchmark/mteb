from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterSemEval2015PC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterSemEval2015",
            "hf_hub_name": "mteb/twittersemeval2015-pairclassification",
            "description": "Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
            "reference": "https://alt.qcri.org/semeval2015/task1/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ap",
            "revision": "70970daeab8776df92f5ea462b6173c0b46fd2d1",
        }
