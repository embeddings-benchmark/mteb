from ...abstasks.AbsTaskReranking import AbsTaskReranking


class AlloprofReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "AlloprofReranking",
            "hf_hub_name": "lyon-nlp/mteb-fr-reranking-alloprof-s2p",
            "description": (
                "This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum"
                "curated by a large number of teachers to students on all subjects taught from in primary and secondary school"
            ),
            "reference": "https://huggingface.co/datasets/antoinelb7/alloprof",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "map",
            "revision": "666fdacebe0291776e86f29345663dfaf80a0db9",
        }
