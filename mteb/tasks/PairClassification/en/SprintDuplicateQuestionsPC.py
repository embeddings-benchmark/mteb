from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SprintDuplicateQuestionsPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "SprintDuplicateQuestions",
            "hf_hub_name": "mteb/sprintduplicatequestions-pairclassification",
            "description": "Duplicate questions from the Sprint community.",
            "reference": "https://www.aclweb.org/anthology/D18-1131/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["en"],
            "main_score": "ap",
            "revision": "d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46",
        }
