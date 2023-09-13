from ...abstasks import AbsTaskClassification


class PatentClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "PatentClassification",
            "hf_hub_name": "jinaai/small_patent",
            "description": "Patent classification evaluation based on the test set of the big patent dataset",
            "reference": "https://dblp.org/rec/journals/corr/abs-1906-03741.bib",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "2b7c7dcba3b82373706ca56d1ab5526eb007e2ee",
        }


class NewsGroupsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NewsGroupsClassification",
            "hf_hub_name": "jinaai/20newsgroups",
            "description": "Category classification evaluation based on the 20 newsgroups dataset",
            "reference": "http://people.csail.mit.edu/jrennie/20Newsgroups/",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "dc56aa588069550b1399f624ee90fc1017515079",
        }
