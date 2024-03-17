from ...abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class RobustReranking(AbsTaskInstructionRetrieval):
    @property
    def description(self):
        return {
            "name": "RobustInstructionRetrieval",
            "hf_hub_name": "jhu-clsp/robust04-instructions",
            "description": "Measuring instruction following ability on Robust04 narratives.",
            "reference": "TODO",
            "type": "InstructionRetrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        }
