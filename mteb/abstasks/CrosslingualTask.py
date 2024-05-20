from __future__ import annotations

from .AbsTask import AbsTask
from .MultiSubsetLoader import MultiSubsetLoader


class CrosslingualTask(MultiSubsetLoader, AbsTask):
    def __init__(self, hf_subsets=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(hf_subsets, list):
            hf_subsets = [
                lang for lang in hf_subsets if lang in self.metadata.eval_langs
            ]
        if hf_subsets is not None and len(hf_subsets) > 0:
            self.hf_subsets = hf_subsets
        else:
            self.hf_subsets = self.metadata.eval_langs
        self.is_crosslingual = True
