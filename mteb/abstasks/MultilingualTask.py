from __future__ import annotations

import datasets

from .AbsTask import AbsTask


class MultilingualTask(AbsTask):
    def __init__(self, langs=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(langs, list):
            langs = [lang for lang in langs if lang in self.metadata_dict["eval_langs"]]
        if langs is not None and len(langs) > 0:
            self.langs = (
                langs  # TODO: case where user provides langs not in the dataset
            )
        else:
            self.langs = self.metadata_dict["eval_langs"]
        self.is_multilingual = True

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                name=lang,
                **self.metadata_dict.get("dataset", None),
            )
        self.dataset_transform()
        self.data_loaded = True
