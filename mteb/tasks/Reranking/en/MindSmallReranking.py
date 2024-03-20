from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MindSmallReranking(AbsTaskReranking):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "mteb/mind_small",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "3bdac13927fdc888b903db93b2ffdbd90b295a69",
        }
