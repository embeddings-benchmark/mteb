from __future__ import annotations

from .LegalBenchPC import LegalBenchPC
from .PubChemAISentenceParaphrasePC import PubChemAISentenceParaphrasePC
from .PubChemSMILESPC import PubChemSMILESPC
from .PubChemSynonymPC import PubChemSynonymPC
from .PubChemWikiParagraphsPC import PubChemWikiParagraphsPC
from .SprintDuplicateQuestionsPC import SprintDuplicateQuestionsPC
from .TwitterSemEval2015PC import TwitterSemEval2015PC
from .TwitterURLCorpusPC import TwitterURLCorpusPC

__all__ = [
    "PubChemSMILESPC",
    "PubChemSynonymPC",
    "LegalBenchPC",
    "TwitterSemEval2015PC",
    "PubChemWikiParagraphsPC",
    "TwitterURLCorpusPC",
    "SprintDuplicateQuestionsPC",
    "PubChemAISentenceParaphrasePC",
]
