from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "SICK-E-PL",
            "hf_hub_name": "PL-MTEB/sicke-pl-pairclassification",
            "description": "Task based on the Polish version of the English natural language inference (NLI) dataset "
                           "SICK (Sentences Involving Compositional Knowledge), where the goal is to predict whether "
                           "the first statement (premise) semantically entails the second statement (hypothesis).",
            "reference": "https://aclanthology.org/2020.lrec-1.207.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap",
        }


class PpcPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "PPC",
            "hf_hub_name": "PL-MTEB/ppc-pairclassification",
            "description": "Task based on the Polish Paraphrase Corpus (PPC), where the goal is to predict whether "
                           "one sentence is a paraphrase of another.",
            "reference": "https://arxiv.org/pdf/2207.12759.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap"
        }


class CdscePC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "CDSC-E",
            "hf_hub_name": "PL-MTEB/cdsce-pairclassification",
            "description": "Task based on the Compositional Distributional Semantics Corpus (CDSC), where the goal is "
                           "to predict whether the first statement (premise) semantically entails the second statement (hypothesis).",
            "reference": "https://aclanthology.org/P17-1073.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap"
        }


class PscPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "PSC",
            "hf_hub_name": "PL-MTEB/psc-pairclassification",
            "description": "The Polish Summaries Corpus contains news articles and their summaries. "
                           "The task is to predict whether the extract text and summary are similar.",
            "reference": "http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap"
        }
