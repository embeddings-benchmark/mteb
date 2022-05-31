from ...abstasks.AbsTaskBinaryClassification import AbsTaskBinaryClassification


class TwitterSemEval2015BC(AbsTaskBinaryClassification):
    @property
    def description(self):
        return {
            "name": "TwitterSemEval2015",
            "hf_hub_name": "mteb/twittersemeval2015-binaryclassification",
            "description": "Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
            "reference": "https://alt.qcri.org/semeval2015/task1/",
            "category": "s2s",
            "type": "BinaryClassification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ap",
        }
