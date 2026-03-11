from .clarqa import ClarQA
from .cremad import CREMADPairClassification
from .legal_bench_pc import LegalBenchPC
from .nmsqa import NMSQAPairClassification
from .pub_chem_ai_sentence_paraphrase_pc import PubChemAISentenceParaphrasePC
from .pub_chem_smilespc import PubChemSMILESPC
from .pub_chem_synonym_pc import PubChemSynonymPC
from .pub_chem_wiki_paragraphs_pc import PubChemWikiParagraphsPC
from .qrecc import QRECC
from .sprint_duplicate_questions_pc import SprintDuplicateQuestionsPC
from .twitter_sem_eval2015_pc import TwitterSemEval2015PC
from .twitter_url_corpus_pc import TwitterURLCorpus
from .vocal_sound import VocalSoundPairClassification
from .vox_populi_accent import VoxPopuliAccentPairClassification

__all__ = [
    "QRECC",
    "CREMADPairClassification",
    "ClarQA",
    "LegalBenchPC",
    "NMSQAPairClassification",
    "PubChemAISentenceParaphrasePC",
    "PubChemSMILESPC",
    "PubChemSynonymPC",
    "PubChemWikiParagraphsPC",
    "SprintDuplicateQuestionsPC",
    "TwitterSemEval2015PC",
    "TwitterURLCorpus",
    "VocalSoundPairClassification",
    "VoxPopuliAccentPairClassification",
]
