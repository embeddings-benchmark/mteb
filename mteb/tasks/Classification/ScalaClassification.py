import datasets

from mteb.abstasks import AbsTaskClassification


class ScalaDaClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaDaClassification",
            "hf_hub_name": "ScandEval/scala-da",
            "description": "A modified version of DDT modified for linguistic acceptability classification",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "1de08520a7b361e92ffa2a2201ebd41942c54675",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])


class ScalaNbClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaNbClassification",
            "hf_hub_name": "ScandEval/scala-nb",
            "description": "A Norwegian dataset for linguistic acceptability classification for Bokmål",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no", "nb"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "237111a078ad5a834a55c57803d40bbe410ed03b",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])


class ScalaNnClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaNbClassification",
            "hf_hub_name": "ScandEval/scala-nn",
            "description": "A Norwegian dataset for linguistic acceptability classification for Nynorsk",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no", "nn"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "9d9a2a4092ed3cacf0744592f6d2f32ab8ef4c0b",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])


class ScalaSvClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaSvClassification",
            "hf_hub_name": "ScandEval/scala-sv",
            "description": "A Swedish dataset for linguistic acceptability classification",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "1b48e3dcb02872335ff985ff938a054a4ed99008",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])
