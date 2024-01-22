from ...abstasks.AbsTaskReranking import AbsTaskReranking

import datasets


class MIRACLRerankingDE(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MIRACL",
            "hf_hub_name": "jinaai/miracl",
            "reference": "https://project-miracl.github.io/",
            "description": (
                "MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual "
                "retrieval dataset that focuses on search across 18 different languages. This task focuses on "
                "the German subset, uing the dev set containing 305 queries."
            ),
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "map",
            "revision": "d28a029f35c4ff7f616df47b0edf54e6882395e6",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            self.description["eval_langs"][0],
            revision=self.description.get("revision", None),
        )
        self.data_loaded = True

class MIRACLRerankingES(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MIRACLES',
            'hf_hub_name': 'jinaai/miracl-es',
            'reference': 'https://project-miracl.github.io/',
            'description': (
                'MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual '
                'retrieval dataset that focuses on search across 18 different languages. This task focuses on '
                'the Spanish subset, using the test set containing 648 queries and 6443 passages.'
            ),
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['es'],
            'main_score': 'map',
            'revision': 'd28a029f35c4ff7f616df47b0edf54e6882395e6',
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            self.description["eval_langs"][0],
            revision=self.description.get("revision", None),
        )
        self.data_loaded = True