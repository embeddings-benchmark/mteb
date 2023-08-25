from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "SICK-E-PL",
            "hf_hub_name": "PL-MTEB/sicke-pl-pairclassification",
            "description": "Polish version of SICK dataset for textual entailment.",
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
            "description": "Polish Paraphrase Corpus",
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
            "description": "Compositional Distributional Semantics Corpus for textual entailment.",
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
            "description": "Polish Summaries Corpus",
            "reference": "http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap"
        }
