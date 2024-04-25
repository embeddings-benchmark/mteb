from __future__ import annotations

import datasets

from .AbsTask import AbsTask


class CrosslingualTask(AbsTask):
    def __init__(self, langs=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(langs, list):
            langs = [lang for lang in langs if lang in self.metadata_dict["eval_langs"]]
        if langs is not None and len(langs) > 0:
            self.langs = langs
        else:
            self.langs = self.metadata_dict["eval_langs"]
        self.is_crosslingual = True

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        fast_loading = self.fast_loading if hasattr(self, "fast_loading") else False
        if fast_loading:
            self.fast_load()
        else:
            self.slow_load()

    def fast_load(self, **kwargs):
        """Load all subsets at once, then group by language with Polars"""
        self.dataset = {}
        merged_dataset = datasets.load_dataset(
            **self.metadata_dict["dataset"]
        )  # load "default" subset
        for split in self.metadata.eval_splits:
            grouped_by_lang = dict(merged_dataset[split].to_polars().group_by("lang"))
            for lang in self.langs:
                if lang not in self.dataset:
                    self.dataset[lang] = dict()
                self.dataset[lang][split] = datasets.Dataset.from_polars(
                    grouped_by_lang[lang].drop("lang")
                )  # Remove lang column and convert back to HF datasets, not strictly necessary but better for compatibility
        self.data_loaded = True

    def slow_load(self, **kwargs):
        """Each subsets is loaded iteratively"""
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                name=lang, **self.metadata_dict["dataset"]
            )
        self.data_loaded = True
