from __future__ import annotations

from .AbsTask import AbsTask
from .MultiSubsetLoader import MultiSubsetLoader


class MultilingualTask(MultiSubsetLoader, AbsTask):
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
