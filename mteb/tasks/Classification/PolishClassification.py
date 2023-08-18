from ...abstasks import AbsTaskClassification


class CbdClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "CBD",
            "hf_hub_name": "PL-MTEB/cbd",
            "description": "The Cyberbullying Detection task where the goal is to predict if a given Twitter message "
                           "contains a cyberbullying (harmful) content.",
            "reference": "http://2019.poleval.pl/files/poleval2019.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy"
        }


class PolEmo2InClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "PolEmo2.0-IN",
            "hf_hub_name": "PL-MTEB/polemo2_in",
            "description": "The PolEmo2.0 is a set of online reviews from four domains: medicine, hotels, products and "
                           "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) "
                           "reviews.",
            "reference": "https://aclanthology.org/K19-1092.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy"
        }


class PolEmo2OutClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "PolEmo2.0-OUT",
            "hf_hub_name": "PL-MTEB/polemo2_out",
            "description": "The PolEmo2.0 is a set of online reviews from four domains: medicine, hotels, products and "
                           "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
                           "school) reviews using models train on reviews from medicine and hotels domains.",
            "reference": "https://aclanthology.org/K19-1092.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy"
        }


class AllegroReviewsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AllegroReviews",
            "hf_hub_name": "PL-MTEB/allegro-reviews",
            "description": "The Allegro Reviews is a set of product reviews from a popular e-commerce marketplace "
                           "(Allegro.pl). The task is to predict a rating ranging from 1 to 5.",
            "reference": "https://aclanthology.org/2020.acl-main.111.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy"
        }


class AbusiveClausesClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AbusiveClauses",
            "hf_hub_name": "laugustyniak/abusive-clauses-pl",
            "description": "Abusive Clauses Detection",
            "reference": "https://arxiv.org/pdf/2211.13112.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "accuracy"
        }
