from .av_speakerbench_pc import AVSpeakerBenchPairClassification
from .ave_dataset_pc import (
    AVEDatasetVAPairClassification,
    AVEDatasetVPairClassification,
)
from .cremad import CREMADPairClassification
from .human_animal_cartoon_pc import (
    HumanAnimalCartoonVAPairClassification,
    HumanAnimalCartoonVPairClassification,
)
from .legal_bench_pc import LegalBenchPC
from .meld_pc import (
    MELDVAPairClassification,
    MELDVPairClassification,
)
from .nmsqa import NMSQAPairClassification
from .pub_chem_ai_sentence_paraphrase_pc import PubChemAISentenceParaphrasePC
from .pub_chem_smilespc import PubChemSMILESPC
from .pub_chem_synonym_pc import PubChemSynonymPC
from .pub_chem_wiki_paragraphs_pc import PubChemWikiParagraphsPC
from .sprint_duplicate_questions_pc import SprintDuplicateQuestionsPC
from .twitter_sem_eval2015_pc import TwitterSemEval2015PC
from .twitter_url_corpus_pc import TwitterURLCorpus
from .videocon_pc import VideoConPairClassification
from .vinoground_pc import VinogroundPairClassification
from .vocal_sound import VocalSoundPairClassification
from .vox_populi_accent import VoxPopuliAccentPairClassification

__all__ = [
    "AVEDatasetVAPairClassification",
    "AVEDatasetVPairClassification",
    "AVSpeakerBenchPairClassification",
    "CREMADPairClassification",
    "HumanAnimalCartoonVAPairClassification",
    "HumanAnimalCartoonVPairClassification",
    "LegalBenchPC",
    "MELDVAPairClassification",
    "MELDVPairClassification",
    "NMSQAPairClassification",
    "PubChemAISentenceParaphrasePC",
    "PubChemSMILESPC",
    "PubChemSynonymPC",
    "PubChemWikiParagraphsPC",
    "SprintDuplicateQuestionsPC",
    "TwitterSemEval2015PC",
    "TwitterURLCorpus",
    "VideoConPairClassification",
    "VinogroundPairClassification",
    "VocalSoundPairClassification",
    "VoxPopuliAccentPairClassification",
]
