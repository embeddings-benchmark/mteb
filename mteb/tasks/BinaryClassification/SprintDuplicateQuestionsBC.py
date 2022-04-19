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
            "available_splits": ["validation", "test"],
            "main_score": "ap",
        }