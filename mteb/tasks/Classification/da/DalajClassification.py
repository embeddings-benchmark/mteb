# SuperLIM tasks
import datasets

from mteb.abstasks import AbsTaskClassification


class DalajClassification(AbsTaskClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "DalajClassification",
            "hf_hub_name": "AI-Sweden/SuperLim",
            "description": "A Swedish dataset for linguistic acceptability. Available as a part of Superlim.",
            "reference": "https://spraakbanken.gu.se/en/resources/superlim",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "dalaj",  # chose the relevant subset
            revision=self.metadata_dict.get("revision"),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        This dataset consist of two columns of relevance, "original_sentence" and "corrected_sentence".
        We will use the original sentence as we "wrong" sentence and the corrected sentence as the "correct" sentence
        """

        def __convert_sample_to_classification(sample):
            text = sample["original_sentence"] + sample["corrected_sentence"]
            label = [1] * len(sample["original_sentence"]) + [0] * len(
                sample["corrected_sentence"]
            )
            return {"text": text, "label": label}

        columns_to_keep = ["original_sentence", "corrected_sentence"]
        for split in self.dataset:
            columns_names = self.dataset[split].column_names  # type: ignore
            columns_to_remove = [
                col for col in columns_names if col not in columns_to_keep
            ]
            self.dataset[split] = self.dataset[split].remove_columns(columns_to_remove)  # type: ignore

        self.dataset = self.dataset.map(
            __convert_sample_to_classification,
            batched=True,
            remove_columns=columns_to_keep,
        )
