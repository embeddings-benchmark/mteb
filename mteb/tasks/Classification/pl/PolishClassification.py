from ....abstasks import AbsTaskClassification


class CbdClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CBD",
            "hf_hub_name": "PL-MTEB/cbd",
            "description": "Polish Tweets annotated for cyberbullying detection.",
            "reference": "http://2019.poleval.pl/files/poleval2019.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy",
        }


class PolEmo2InClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "PolEmo2.0-IN",
            "hf_hub_name": "PL-MTEB/polemo2_in",
            "description": "A collection of Polish online reviews from four domains: medicine, hotels, products and "
            "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) "
            "reviews.",
            "reference": "https://aclanthology.org/K19-1092.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy",
        }


class PolEmo2OutClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "PolEmo2.0-OUT",
            "hf_hub_name": "PL-MTEB/polemo2_out",
            "description": "A collection of Polish online reviews from four domains: medicine, hotels, products and "
            "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
            "school) reviews using models train on reviews from medicine and hotels domains.",
            "reference": "https://aclanthology.org/K19-1092.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy",
        }


class AllegroReviewsClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "AllegroReviews",
            "hf_hub_name": "PL-MTEB/allegro-reviews",
            "description": "A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.",
            "reference": "https://aclanthology.org/2020.acl-main.111.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy",
        }


class PacClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "PAC",
            "hf_hub_name": "laugustyniak/abusive-clauses-pl",
            "description": "Polish Abusive Clauses Dataset",
            "reference": "https://arxiv.org/pdf/2211.13112.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy",
        }
