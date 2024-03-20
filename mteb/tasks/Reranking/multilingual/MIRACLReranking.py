from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MIRACLReranking(MultilingualTask, AbsTaskReranking):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MIRACLReranking",
            "hf_hub_name": "jinaai/miracl",
            "reference": "https://project-miracl.github.io/",
            "description": (
                "MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual "
                "retrieval dataset that focuses on search across 18 different languages. This task focuses on "
                "the German and Spanish subset."
            ),
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["de", "es"],
            "main_score": "map",
            "revision": "d28a029f35c4ff7f616df47b0edf54e6882395e6",
        }
