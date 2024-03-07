from ...abstasks.AbsTaskInstructionReranking import AbsTaskInstructionReranking


class RobustReranking(AbsTaskInstructionReranking):
    @property
    def description(self):
        return {
            "name": "RobustReranking",
            "hf_hub_name": "mteb/scidocs-reranking",
            "description": "Measuring instruction following ability on Robust04 narratives.",
            "reference": "TODO",
            "type": "InstructionReranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        }
