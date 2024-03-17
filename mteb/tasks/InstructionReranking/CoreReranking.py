from ...abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class NewsReranking(AbsTaskInstructionRetrieval):
    @property
    def description(self):
        return {
            "name": "CoreInstructionRetrieval",
            "hf_hub_name": "jhu-clsp/core17-instructions",
            "description": "Measuring instruction following ability on TREC Core 17 narratives.",
            "reference": "TODO",
            "type": "InstructionRetrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        }
