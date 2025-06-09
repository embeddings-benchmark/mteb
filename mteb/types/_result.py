from __future__ import annotations

from typing import Any

HFSubset = str # e.g. 'en-de', 'en', 'default' (default is used when there is no subset)
SplitName = str # e.g. 'test', 'validation', 'train'
Score = Any # e.g. float, int, or any other type that represents a score, should be json serializable
ScoresDict = dict[str, Any] # e.g {'main_score': 0.5, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']}
