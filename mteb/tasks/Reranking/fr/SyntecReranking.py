from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SyntecReranking(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SyntecReranking",
            "hf_hub_name": "lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "description": "This dataset has been built from the Syntec Collective bargaining agreement.",
            "reference": "https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "map",
            "revision": "b205c5084a0934ce8af14338bf03feb19499c84d",
        }
