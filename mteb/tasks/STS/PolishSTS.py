from ...abstasks.AbsTaskSTS import AbsTaskSTS


class SickrPLSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "SICK-R-PL",
            "hf_hub_name": "PL-MTEB/sickr-pl-sts",
            "description": "Task based on the Polish version of the English natural language inference (NLI) dataset "
                           "SICK (Sentences Involving Compositional Knowledge), where the goal is to predict the "
                           "probability distribution of relatedness scores between statements",
            "reference": "https://aclanthology.org/2020.lrec-1.207.pdf",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5
        }


class CdscrSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "CDSC-R",
            "hf_hub_name": "PL-MTEB/cdscr-sts",
            "description": "Task based on the Compositional Distributional Semantics Corpus (CDSC), where the goal is to "
                           "predict the probability distribution of relatedness scores between statements.",
            "reference": "https://aclanthology.org/P17-1073.pdf",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5
        }