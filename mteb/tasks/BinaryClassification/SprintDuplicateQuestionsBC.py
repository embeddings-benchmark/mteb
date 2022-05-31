from ...abstasks.AbsTaskBinaryClassification import AbsTaskBinaryClassification


class SprintDuplicateQuestionsBC(AbsTaskBinaryClassification):
    @property
    def description(self):
        return {
            "name": "SprintDuplicateQuestions",
            "hf_hub_name": "mteb/sprintduplicatequestions-binaryclassification",
            "description": "Duplicate questions from the Sprint community.",
            "reference": "https://www.aclweb.org/anthology/D18-1131/",
            "category": "s2s",
            "type": "BinaryClassification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["en"],
            "main_score": "ap",
        }
