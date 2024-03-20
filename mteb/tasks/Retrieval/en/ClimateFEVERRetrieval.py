from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ClimateFEVER(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "ClimateFEVER",
            "hf_hub_name": "mteb/climate-fever",
            "description": (
                "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims"
                " regarding climate-change."
            ),
            "reference": "https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "47f2ac6acb640fc46020b02a5b59fdda04d39380",
        }
