from __future__ import annotations

import warnings

from .AbsTask import AbsTask
from .MultiSubsetLoader import MultiSubsetLoader


class MultilingualTask(MultiSubsetLoader, AbsTask):
    def __init__(self, hf_subsets: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        warnings.warn(
            "`MultilingualTask` will be removed in v2.0. In the future, checking whether a task is multilingual"
            " will be based solely on `metadata.eval_langs`, which should be a dictionary for multilingual tasks.",
            DeprecationWarning,
        )
        if isinstance(hf_subsets, list):
            hf_subsets = [
                lang for lang in hf_subsets if lang in self.metadata.eval_langs
            ]
        if hf_subsets is not None and len(hf_subsets) > 0:
            self.hf_subsets = hf_subsets
        else:
            self.hf_subsets = self.metadata.eval_langs
        self.is_multilingual = True
