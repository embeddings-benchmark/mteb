from ....abstasks import AbsTaskClassification


class AmazonPolarityClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "AmazonPolarityClassification",
            "hf_hub_name": "mteb/amazon_polarity",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "e2d317d38cd51312af73b3d32a06d1a08b442046",
        }
