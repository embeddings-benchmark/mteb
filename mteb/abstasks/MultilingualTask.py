from __future__ import annotations

from .AbsTask import AbsTask
from .MultiSubsetLoader import MultiSubsetLoader


class MultilingualTask(MultiSubsetLoader, AbsTask):
    def __init__(self, hf_subsets=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(hf_subsets, list):
            hf_subsets = [
                lang for lang in hf_subsets if lang in self.metadata_dict["eval_langs"]
            ]
        if hf_subsets is not None and len(hf_subsets) > 0:
            self.hf_subsets = (
                hf_subsets  # TODO: case where user provides langs not in the dataset
            )
        else:
            self.hf_subsets = self.metadata_dict["eval_langs"]
        self.is_multilingual = True
