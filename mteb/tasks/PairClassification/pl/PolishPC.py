from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
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
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "PPC",
            "hf_hub_name": "PL-MTEB/ppc-pairclassification",
            "description": "Polish Paraphrase Corpus",
            "reference": "https://arxiv.org/pdf/2207.12759.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap",
        }


class CdscePC(AbsTaskPairClassification):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CDSC-E",
            "hf_hub_name": "PL-MTEB/cdsce-pairclassification",
            "description": "Compositional Distributional Semantics Corpus for textual entailment.",
            "reference": "https://aclanthology.org/P17-1073.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap",
        }


class PscPC(AbsTaskPairClassification):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "PSC",
            "hf_hub_name": "PL-MTEB/psc-pairclassification",
            "description": "Polish Summaries Corpus",
            "reference": "http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ap",
        }
