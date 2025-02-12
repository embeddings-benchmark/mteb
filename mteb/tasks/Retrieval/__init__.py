from __future__ import annotations

from .ara import SadeemQuestionRetrieval
from .code import (
    AppsRetrieval,
    CodeEditSearchRetrieval,
    CodeFeedbackMT,
    CodeFeedbackST,
    CodeRAGLibraryDocumentationSolutionsRetrieval,
    CodeRAGOnlineTutorialsRetrieval,
    CodeRAGProgrammingSolutionsRetrieval,
    CodeRAGStackoverflowPostsRetrieval,
    CodeSearchNetCCRetrieval,
    CodeSearchNetRetrieval,
    CodeTransOceanContestRetrieval,
    CodeTransOceanDLRetrieval,
    COIRCodeSearchNetRetrieval,
    CosQARetrieval,
    StackOverflowQARetrieval,
    SyntheticText2SQLRetrieval,
)
from .dan import DanFever, DanFeverRetrieval, TV2Nordretrieval, TwitterHjerneRetrieval
from .deu import (
    GerDaLIR,
    GerDaLIRSmall,
    GermanDPR,
    GermanGovServiceRetrieval,
    GermanQuADRetrieval,
    LegalQuAD,
)
from .ell import GreekCivicsQA
from .eng import (
    FEVER,
    MSMARCO,
    NQ,
    PIQA,
    SCIDOCS,
    SIQA,
    TRECCOVID,
    AILACasedocs,
    AILAStatutes,
    AlphaNLI,
    Apple10KRetrieval,
    ARCChallenge,
    ArguAna,
    BrightRetrieval,
    ChemHotpotQARetrieval,
    ChemNQRetrieval,
    ClimateFEVER,
    ClimateFEVERHardNegatives,
    CQADupstackAndroidRetrieval,
    CQADupstackEnglishRetrieval,
    CQADupstackGamingRetrieval,
    CQADupstackGisRetrieval,
    CQADupstackMathematicaRetrieval,
    CQADupstackPhysicsRetrieval,
    CQADupstackProgrammersRetrieval,
    CQADupstackStatsRetrieval,
    CQADupstackTexRetrieval,
    CQADupstackUnixRetrieval,
    CQADupstackWebmastersRetrieval,
    CQADupstackWordpressRetrieval,
    DBPedia,
    DBPediaHardNegatives,
    FaithDialRetrieval,
    FeedbackQARetrieval,
    FEVERHardNegatives,
    FinanceBenchRetrieval,
    FinQARetrieval,
    FiQA2018,
    HagridRetrieval,
    HC3Retrieval,
    HellaSwag,
    HotpotQA,
    HotpotQAHardNegatives,
    LegalBenchConsumerContractsQA,
    LegalBenchCorporateLobbying,
    LegalSummarization,
    LEMBNarrativeQARetrieval,
    LEMBNeedleRetrieval,
    LEMBPasskeyRetrieval,
    LEMBQMSumRetrieval,
    LEMBSummScreenFDRetrieval,
    LEMBWikimQARetrieval,
    LitSearchRetrieval,
    MedicalQARetrieval,
    MLQuestionsRetrieval,
    MSMARCOHardNegatives,
    MSMARCOv2,
    NanoArguAnaRetrieval,
    NanoClimateFeverRetrieval,
    NanoDBPediaRetrieval,
    NanoFEVERRetrieval,
    NanoFiQA2018Retrieval,
    NanoHotpotQARetrieval,
    NanoMSMARCORetrieval,
    NanoNFCorpusRetrieval,
    NanoNQRetrieval,
    NanoQuoraRetrieval,
    NanoSCIDOCSRetrieval,
    NanoSciFactRetrieval,
    NanoTouche2020Retrieval,
    NarrativeQARetrieval,
    NFCorpus,
    NQHardNegatives,
    Quail,
    QuoraRetrieval,
    QuoraRetrievalHardNegatives,
    RARbCode,
    RARbMath,
    SciFact,
    SpartQA,
    TATQARetrieval,
    TempReasonL1,
    TempReasonL2Context,
    TempReasonL2Fact,
    TempReasonL2Pure,
    TempReasonL3Context,
    TempReasonL3Fact,
    TempReasonL3Pure,
    TheGoldmanEnRetrieval,
    TopiOCQARetrieval,
    TopiOCQARetrievalHardNegatives,
    Touche2020,
    Touche2020v3Retrieval,
    TradeTheEventEncyclopediaRetrieval,
    TradeTheEventNewsRetrieval,
    USNewsRetrieval,
    WinoGrande,
)
from .est import EstQA
from .fas import (
    ArguAnaFa,
    ClimateFEVERFa,
    CQADupstackAndroidRetrievalFa,
    CQADupstackEnglishRetrievalFa,
    CQADupstackGamingRetrievalFa,
    CQADupstackGisRetrievalFa,
    CQADupstackMathematicaRetrievalFa,
    CQADupstackPhysicsRetrievalFa,
    CQADupstackProgrammersRetrievalFa,
    CQADupstackStatsRetrievalFa,
    CQADupstackTexRetrievalFa,
    CQADupstackUnixRetrievalFa,
    CQADupstackWebmastersRetrievalFa,
    CQADupstackWordpressRetrievalFa,
    DBPediaFa,
    FiQA2018Fa,
    HotpotQAFa,
    MSMARCOFa,
    NFCorpusFa,
    NQFa,
    PersianWebDocumentRetrieval,
    QuoraRetrievalFa,
    SCIDOCSFa,
    SciFactFa,
    SynPerChatbotRAGFAQRetrieval,
    SynPerChatbotRAGTopicsRetrieval,
    SynPerChatbotTopicsRetrieval,
    SynPerQARetrieval,
    Touche2020Fa,
    TRECCOVIDFa,
)
from .fra import AlloprofRetrieval, BSARDRetrieval, FQuADRetrieval, SyntecRetrieval
from .hun import HunSum2AbstractiveRetrieval
from .jpn import (
    JaGovFaqsRetrieval,
    JaqketRetrieval,
    JaQuADRetrieval,
    NLPJournalAbsIntroRetrieval,
    NLPJournalTitleAbsRetrieval,
    NLPJournalTitleIntroRetrieval,
)
from .kat import GeorgianFAQRetrieval
from .kor import AutoRAGRetrieval, KoStrategyQA
from .multilingual import (
    BelebeleRetrieval,
    CrossLingualSemanticDiscriminationWMT19,
    CrossLingualSemanticDiscriminationWMT21,
    CUREv1Retrieval,
    IndicQARetrieval,
    MintakaRetrieval,
    MIRACLRetrieval,
    MIRACLRetrievalHardNegatives,
    MLQARetrieval,
    MrTidyRetrieval,
    MultiLongDocRetrieval,
    NeuCLIR2022Retrieval,
    NeuCLIR2022RetrievalHardNegatives,
    NeuCLIR2023Retrieval,
    NeuCLIR2023RetrievalHardNegatives,
    PublicHealthQARetrieval,
    StatcanDialogueDatasetRetrieval,
    WikipediaRetrievalMultilingual,
    XMarket,
    XPQARetrieval,
    XQuADRetrieval,
)
from .nld import (
    FEVERNL,
    MMMARCONL,
    NQNL,
    SCIDOCSNL,
    TRECCOVIDNL,
    ArguAnaNL,
    ClimateFEVERNL,
    CQADupstackAndroidNLRetrieval,
    CQADupstackEnglishNLRetrieval,
    CQADupstackGamingNLRetrieval,
    CQADupstackGisNLRetrieval,
    CQADupstackMathematicaNLRetrieval,
    CQADupstackPhysicsNLRetrieval,
    CQADupstackProgrammersNLRetrieval,
    CQADupstackStatsNLRetrieval,
    CQADupstackTexNLRetrieval,
    CQADupstackUnixNLRetrieval,
    CQADupstackWebmastersNLRetrieval,
    CQADupstackWordpressNLRetrieval,
    DBPediaNL,
    FiQA2018NL,
    HotpotQANL,
    NFCorpusNL,
    QuoraNLRetrieval,
    SciFactNL,
    Touche2020NL,
)
from .nob import NorQuadRetrieval, SNLRetrieval
from .pol import (
    MSMARCOPL,
    NQPL,
    SCIDOCSPL,
    TRECCOVIDPL,
    ArguAnaPL,
    CQADupstackAndroidRetrievalPL,
    CQADupstackEnglishRetrievalPL,
    CQADupstackGamingRetrievalPL,
    CQADupstackGisRetrievalPL,
    CQADupstackMathematicaRetrievalPL,
    CQADupstackPhysicsRetrievalPL,
    CQADupstackProgrammersRetrievalPL,
    CQADupstackStatsRetrievalPL,
    CQADupstackTexRetrievalPL,
    CQADupstackUnixRetrievalPL,
    CQADupstackWebmastersRetrievalPL,
    CQADupstackWordpressRetrievalPL,
    DBPediaPL,
    DBPediaPLHardNegatives,
    FiQAPLRetrieval,
    HotpotQAPL,
    HotpotQAPLHardNegatives,
    MSMARCOPLHardNegatives,
    NFCorpusPL,
    NQPLHardNegatives,
    QuoraPLRetrieval,
    QuoraPLRetrievalHardNegatives,
    SciFactPL,
    Touche2020PL,
)
from .rus import RiaNewsRetrieval, RiaNewsRetrievalHardNegatives, RuBQRetrieval
from .slk import SKQuadRetrieval, SlovakSumRetrieval
from .spa import SpanishPassageRetrievalS2P, SpanishPassageRetrievalS2S
from .swe import SwednRetrieval, SweFaqRetrieval
from .tur import TurHistQuadRetrieval
from .vie import VieQuADRetrieval
from .zho import (
    AlphaFinRetrieval,
    CmedqaRetrieval,
    CovidRetrieval,
    DISCFinLLMComputingRetrieval,
    DISCFinLLMRetrieval,
    DuEEFinRetrieval,
    DuRetrieval,
    EcomRetrieval,
    FinEvaEncyclopediaRetrieval,
    FinEvaRetrieval,
    FinTruthQARetrieval,
    LeCaRDv2,
    MedicalRetrieval,
    MMarcoRetrieval,
    SmoothNLPRetrieval,
    T2Retrieval,
    TheGoldmanZhRetrieval,
    THUCNewsRetrieval,
    VideoRetrieval,
)

__all__ = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MMarcoRetrieval",
    "MedicalRetrieval",
    "T2Retrieval",
    "VideoRetrieval",
    "LeCaRDv2",
    "SpanishPassageRetrievalS2S",
    "SpanishPassageRetrievalS2P",
    "MSMARCOPL",
    "MSMARCOPLHardNegatives",
    "SCIDOCSPL",
    "SciFactPL",
    "CQADupstackAndroidRetrievalPL",
    "CQADupstackEnglishRetrievalPL",
    "CQADupstackGamingRetrievalPL",
    "CQADupstackGisRetrievalPL",
    "CQADupstackMathematicaRetrievalPL",
    "CQADupstackPhysicsRetrievalPL",
    "CQADupstackProgrammersRetrievalPL",
    "CQADupstackStatsRetrievalPL",
    "CQADupstackTexRetrievalPL",
    "CQADupstackUnixRetrievalPL",
    "CQADupstackWebmastersRetrievalPL",
    "CQADupstackWordpressRetrievalPL",
    "ArguAnaPL",
    "FiQAPLRetrieval",
    "NFCorpusPL",
    "QuoraPLRetrieval",
    "QuoraPLRetrievalHardNegatives",
    "TRECCOVIDPL",
    "NQPL",
    "NQPLHardNegatives",
    "DBPediaPL",
    "DBPediaPLHardNegatives",
    "HotpotQAPL",
    "HotpotQAPLHardNegatives",
    "GeorgianFAQRetrieval",
    "SwednRetrieval",
    "SweFaqRetrieval",
    "SlovakSumRetrieval",
    "SKQuadRetrieval",
    "SNLRetrieval",
    "NorQuadRetrieval",
    "GermanQuADRetrieval",
    "GerDaLIRSmall",
    "GermanDPR",
    "GermanGovServiceRetrieval",
    "LegalQuAD",
    "GerDaLIR",
    "SadeemQuestionRetrieval",
    "TurHistQuadRetrieval",
    "VieQuADRetrieval",
    "CQADupstackEnglishNLRetrieval",
    "DBPediaNL",
    "FiQA2018NL",
    "CQADupstackUnixNLRetrieval",
    "ClimateFEVERNL",
    "CQADupstackPhysicsNLRetrieval",
    "FEVERNL",
    "CQADupstackMathematicaNLRetrieval",
    "MMMARCONL",
    "Touche2020NL",
    "CQADupstackWordpressNLRetrieval",
    "CQADupstackAndroidNLRetrieval",
    "CQADupstackGisNLRetrieval",
    "TRECCOVIDNL",
    "CQADupstackProgrammersNLRetrieval",
    "NFCorpusNL",
    "CQADupstackStatsNLRetrieval",
    "NQNL",
    "CQADupstackGamingNLRetrieval",
    "ArguAnaNL",
    "SCIDOCSNL",
    "HotpotQANL",
    "QuoraNLRetrieval",
    "CQADupstackTexNLRetrieval",
    "CQADupstackWebmastersNLRetrieval",
    "SciFactNL",
    "DanFever",
    "DanFeverRetrieval",
    "TV2Nordretrieval",
    "TwitterHjerneRetrieval",
    "EstQA",
    "Quail",
    "Touche2020",
    "Touche2020v3Retrieval",
    "TempReasonL2Pure",
    "LegalSummarization",
    "NQ",
    "NQHardNegatives",
    "SIQA",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "DBPedia",
    "DBPediaHardNegatives",
    "NarrativeQARetrieval",
    "MSMARCOv2",
    "CQADupstackTexRetrieval",
    "TRECCOVID",
    "WinoGrande",
    "QuoraRetrieval",
    "QuoraRetrievalHardNegatives",
    "AlphaNLI",
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "CQADupstackAndroidRetrieval",
    "TempReasonL2Context",
    "NanoDBPediaRetrieval",
    "ARCChallenge",
    "ChemHotpotQARetrieval",
    "LegalBenchCorporateLobbying",
    "SCIDOCS",
    "MedicalQARetrieval",
    "RARbCode",
    "LEMBQMSumRetrieval",
    "TempReasonL3Context",
    "AILAStatutes",
    "TopiOCQARetrieval",
    "TopiOCQARetrievalHardNegatives",
    "ClimateFEVER",
    "ClimateFEVERHardNegatives",
    "CQADupstackWordpressRetrieval",
    "CQADupstackEnglishRetrieval",
    "NanoTouche2020Retrieval",
    "CQADupstackStatsRetrieval",
    "MLQuestionsRetrieval",
    "TempReasonL2Fact",
    "NanoSciFactRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackWebmastersRetrieval",
    "NanoFiQA2018Retrieval",
    "CQADupstackUnixRetrieval",
    "TempReasonL3Pure",
    "CQADupstackPhysicsRetrieval",
    "FiQA2018",
    "LitSearchRetrieval",
    "NanoFEVERRetrieval",
    "NanoMSMARCORetrieval",
    "FeedbackQARetrieval",
    "HagridRetrieval",
    "NanoNFCorpusRetrieval",
    "FaithDialRetrieval",
    "SciFact",
    "CQADupstackMathematicaRetrieval",
    "RARbMath",
    "NanoNQRetrieval",
    "HellaSwag",
    "PIQA",
    "SpartQA",
    "BrightRetrieval",
    "TempReasonL1",
    "HotpotQA",
    "HotpotQAHardNegatives",
    "NanoClimateFeverRetrieval",
    "NanoQuoraRetrieval",
    "NanoArguAnaRetrieval",
    "LegalBenchConsumerContractsQA",
    "NanoHotpotQARetrieval",
    "ArguAna",
    "LEMBWikimQARetrieval",
    "TempReasonL3Fact",
    "FEVER",
    "FEVERHardNegatives",
    "CQADupstackGisRetrieval",
    "NanoSCIDOCSRetrieval",
    "AILACasedocs",
    "NFCorpus",
    "ChemNQRetrieval",
    "LEMBSummScreenFDRetrieval",
    "LEMBNarrativeQARetrieval",
    "CQADupstackProgrammersRetrieval",
    "PersianWebDocumentRetrieval",
    "SynPerChatbotRAGFAQRetrieval",
    "SynPerChatbotRAGTopicsRetrieval",
    "SynPerChatbotTopicsRetrieval",
    "SynPerQARetrieval",
    "ArguAnaFa",
    "CQADupstackAndroidRetrievalFa",
    "CQADupstackEnglishRetrievalFa",
    "CQADupstackGamingRetrievalFa",
    "CQADupstackGisRetrievalFa",
    "CQADupstackMathematicaRetrievalFa",
    "CQADupstackPhysicsRetrievalFa",
    "CQADupstackProgrammersRetrievalFa",
    "CQADupstackStatsRetrievalFa",
    "CQADupstackTexRetrievalFa",
    "CQADupstackUnixRetrievalFa",
    "CQADupstackWebmastersRetrievalFa",
    "CQADupstackWordpressRetrievalFa",
    "ClimateFEVERFa",
    "DBPediaFa",
    "FiQA2018Fa",
    "HotpotQAFa",
    "MSMARCOFa",
    "NFCorpusFa",
    "NQFa",
    "QuoraRetrievalFa",
    "SCIDOCSFa",
    "SciFactFa",
    "TRECCOVIDFa",
    "Touche2020Fa",
    "JaGovFaqsRetrieval",
    "NLPJournalAbsIntroRetrieval",
    "JaqketRetrieval",
    "NLPJournalTitleAbsRetrieval",
    "JaQuADRetrieval",
    "NLPJournalTitleIntroRetrieval",
    "HunSum2AbstractiveRetrieval",
    "AutoRAGRetrieval",
    "KoStrategyQA",
    "WikipediaRetrievalMultilingual",
    "MintakaRetrieval",
    "PublicHealthQARetrieval",
    "CrossLingualSemanticDiscriminationWMT19",
    "MultiLongDocRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "StatcanDialogueDatasetRetrieval",
    "IndicQARetrieval",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2023RetrievalHardNegatives",
    "CrossLingualSemanticDiscriminationWMT21",
    "XMarket",
    "XPQARetrieval",
    "BelebeleRetrieval",
    "CUREv1Retrieval",
    "MLQARetrieval",
    "XQuADRetrieval",
    "MrTidyRetrieval",
    "CodeTransOceanContestRetrieval",
    "CodeTransOceanDLRetrieval",
    "CodeFeedbackMT",
    "CodeRAGLibraryDocumentationSolutionsRetrieval",
    "CodeRAGOnlineTutorialsRetrieval",
    "CodeRAGProgrammingSolutionsRetrieval",
    "CodeRAGStackoverflowPostsRetrieval",
    "CodeSearchNetCCRetrieval",
    "StackOverflowQARetrieval",
    "CodeFeedbackST",
    "CosQARetrieval",
    "CodeEditSearchRetrieval",
    "SyntheticText2SQLRetrieval",
    "AppsRetrieval",
    "CodeSearchNetRetrieval",
    "COIRCodeSearchNetRetrieval",
    "RiaNewsRetrieval",
    "RiaNewsRetrievalHardNegatives",
    "RuBQRetrieval",
    "GreekCivicsQA",
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "SyntecRetrieval",
    "FQuADRetrieval",
    "Touche2020PL",
    "Apple10KRetrieval",
    "FinQARetrieval",
    "FinanceBenchRetrieval",
    "HC3Retrieval",
    "TATQARetrieval",
    "TheGoldmanEnRetrieval",
    "TradeTheEventEncyclopediaRetrieval",
    "TradeTheEventNewsRetrieval",
    "USNewsRetrieval",
    "AlphaFinRetrieval",
    "DISCFinLLMComputingRetrieval",
    "DISCFinLLMRetrieval",
    "DuEEFinRetrieval",
    "FinEvaEncyclopediaRetrieval",
    "FinEvaRetrieval",
    "FinTruthQARetrieval",
    "SmoothNLPRetrieval",
    "THUCNewsRetrieval",
    "TheGoldmanZhRetrieval",
]
