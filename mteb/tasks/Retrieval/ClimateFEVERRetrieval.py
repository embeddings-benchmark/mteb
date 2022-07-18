from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class ClimateFEVER(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "ClimateFEVER",
            "beir_name": "climate-fever",
            "description": (
                "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims"
                " regarding climate-change."
            ),
            "reference": "https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
