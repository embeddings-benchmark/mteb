from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraPLRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "Quora-PL",
            "hf_hub_name": "clarin-knext/quora-pl",
            "description": (
                "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
                " question, find other (duplicate) questions."
            ),
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "type": "Retrieval",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "category": "s2s",
            "eval_splits": ["validation", "test"],  # validation for new DataLoader
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "0be27e93455051e531182b85e85e425aba12e9d4",
        }
