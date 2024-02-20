from ...abstasks.AbsTaskSTS import AbsTaskSTS
import datasets


class SickFrSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "SICKFr",
            "hf_hub_name": "Lajavaness/SICK-fr",
            "description": "SICK dataset french version",
            "reference": "https://huggingface.co/datasets/Lajavaness/SICK-fr",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["fr"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
            "revision": "e077ab4cf4774a1e36d86d593b150422fafd8e8a",
        }
    
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and rename columns to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )

        self.dataset = self.dataset.rename_columns({
        "sentence_A": "sentence1",  "sentence_B": "sentence2", 
        "relatedness_score": "score", "Unnamed: 0": "id"
        })
        self.data_loaded = True
