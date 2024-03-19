from ....abstasks.AbsTaskReranking import AbsTaskReranking


class AskUbuntuDupQuestions(AbsTaskReranking):
    @property
    def metadata_dict(self):
        return {
            "name": "AskUbuntuDupQuestions",
            "hf_hub_name": "mteb/askubuntudupquestions-reranking",
            "description": (
                "AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of"
                " questions as similar or non-similar"
            ),
            "reference": "https://github.com/taolei87/askubuntu",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "2000358ca161889fa9c082cb41daa8dcfb161a54",
        }
