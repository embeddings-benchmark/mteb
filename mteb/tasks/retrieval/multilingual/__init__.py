from .afri_mcqa_retrieval import AfriMCQAA2IRetrieval, AfriMCQAI2ARetrieval
from .audio_caps import AudioCapsA2TRetrieval, AudioCapsT2ARetrieval
from .belebele_retrieval import BelebeleRetrieval
from .common_voice import (
    CommonVoiceMini17A2TRetrieval,
    CommonVoiceMini17T2ARetrieval,
    CommonVoiceMini21A2TRetrieval,
    CommonVoiceMini21T2ARetrieval,
)
from .cross_lingual_semantic_discrimination_wmt19 import (
    CrossLingualSemanticDiscriminationWMT19,
)
from .cross_lingual_semantic_discrimination_wmt21 import (
    CrossLingualSemanticDiscriminationWMT21,
)
from .cur_ev1_retrieval import CUREv1Retrieval
from .euro_pirq_retrieval import EuroPIRQRetrieval
from .fleurs import (
    FleursA2TRetrieval,
    FleursA2TRetrievalV2,
    FleursT2ARetrieval,
    FleursT2ARetrievalV2,
)
from .glami_1m_retrieval import GLAMI1MI2TRetrieval, GLAMI1MT2IRetrieval
from .google_svq import GoogleSVQA2TRetrieval, GoogleSVQT2ARetrieval
from .indic_qa_retrieval import IndicQARetrieval
from .jam_alt import (
    JamAltArtistA2ARetrieval,
    JamAltLyricA2TRetrieval,
    JamAltLyricT2ARetrieval,
)
from .jina_vdr_bench_retrieval import (
    JinaVDRAirbnbSyntheticRetrieval,
    JinaVDRArabicChartQARetrieval,
    JinaVDRArabicInfographicsVQARetrieval,
    JinaVDRArxivQARetrieval,
    JinaVDRAutomobileCatelogRetrieval,
    JinaVDRBeveragesCatalogueRetrieval,
    JinaVDRChartQARetrieval,
    JinaVDRCharXivOCRRetrieval,
    JinaVDRDocQAAI,
    JinaVDRDocQAEnergyRetrieval,
    JinaVDRDocQAGovReportRetrieval,
    JinaVDRDocQAHealthcareIndustryRetrieval,
    JinaVDRDocVQARetrieval,
    JinaVDRDonutVQAISynHMPRetrieval,
    JinaVDREuropeanaDeNewsRetrieval,
    JinaVDREuropeanaEsNewsRetrieval,
    JinaVDREuropeanaFrNewsRetrieval,
    JinaVDREuropeanaItScansRetrieval,
    JinaVDREuropeanaNlLegalRetrieval,
    JinaVDRGitHubReadmeRetrieval,
    JinaVDRHindiGovVQARetrieval,
    JinaVDRHungarianDocQARetrieval,
    JinaVDRInfovqaRetrieval,
    JinaVDRJDocQARetrieval,
    JinaVDRJina2024YearlyBookRetrieval,
    JinaVDRMedicalPrescriptionsRetrieval,
    JinaVDRMMTabRetrieval,
    JinaVDRMPMQARetrieval,
    JinaVDROpenAINewsRetrieval,
    JinaVDROWIDChartsRetrieval,
    JinaVDRPlotQARetrieval,
    JinaVDRRamensBenchmarkRetrieval,
    JinaVDRShanghaiMasterPlanRetrieval,
    JinaVDRShiftProjectRetrieval,
    JinaVDRStanfordSlideRetrieval,
    JinaVDRStudentEnrollmentSyntheticRetrieval,
    JinaVDRTabFQuadRetrieval,
    JinaVDRTableVQARetrieval,
    JinaVDRTatQARetrieval,
    JinaVDRTQARetrieval,
    JinaVDRTweetStockSyntheticsRetrieval,
    JinaVDRWikimediaCommonsDocumentsRetrieval,
    JinaVDRWikimediaCommonsMapsRetrieval,
)
from .mintaka_retrieval import MintakaRetrieval
from .miracl_retrieval import (
    MIRACLRetrieval,
    MIRACLRetrievalHardNegatives,
    MIRACLRetrievalHardNegativesV2,
)
from .miracl_vision_retrieval import MIRACLVisionRetrieval
from .mkqa_retrieval import MKQARetrieval
from .mlqa_retrieval import MLQARetrieval
from .mmarco_retrieval import MMarcoRetrievalMultilingual
from .mr_tidy_retrieval import MrTidyRetrieval
from .multi30k_retrieval import Multi30kI2TRetrieval, Multi30kT2IRetrieval
from .multi_long_doc_retrieval import MultiLongDocRetrieval
from .mupler_retrieval import MuPLeRRetrieval
from .nanobeir_multilingual import (
    MultilingualNanoArguAnaRetrieval,
    MultilingualNanoClimateFeverRetrieval,
    MultilingualNanoDBPediaRetrieval,
    MultilingualNanoFEVERRetrieval,
    MultilingualNanoFiQA2018Retrieval,
    MultilingualNanoHotpotQARetrieval,
    MultilingualNanoMSMARCORetrieval,
    MultilingualNanoNFCorpusRetrieval,
    MultilingualNanoNQRetrieval,
    MultilingualNanoQuoraRetrieval,
    MultilingualNanoSCIDOCSRetrieval,
    MultilingualNanoSciFactRetrieval,
    MultilingualNanoTouche2020Retrieval,
)
from .neu_clir2022_retrieval import (
    NeuCLIR2022Retrieval,
    NeuCLIR2022RetrievalHardNegatives,
)
from .neu_clir2023_retrieval import (
    NeuCLIR2023Retrieval,
    NeuCLIR2023RetrievalHardNegatives,
)
from .news_retrieval import GlobalNewsRetrieval, PublicNewsRetrieval
from .omnilingual_asr_retrieval import (
    OmnilingualASRA2TRetrieval,
    OmnilingualASRT2ARetrieval,
)
from .public_health_qa_retrieval import PublicHealthQARetrieval
from .ru_sci_bench_retrieval import RuSciBenchCiteRetrieval, RuSciBenchCociteRetrieval
from .spoken_wikipedia_retrieval import (
    SpokenWikipediaA2TRetrieval,
    SpokenWikipediaT2ARetrieval,
)
from .statcan_dialogue_dataset_retrieval import StatcanDialogueDatasetRetrieval
from .vaani_speech_text_retrieval import VaaniA2TRetrieval, VaaniT2ARetrieval
from .vdr_multilingual_retrieval import VDRMultilingualRetrieval
from .vidore2_bench_retrieval import (
    Vidore2BioMedicalLecturesRetrieval,
    Vidore2EconomicsReportsRetrieval,
    Vidore2ESGReportsHLRetrieval,
    Vidore2ESGReportsRetrieval,
)
from .vidore3_bench_retrieval import (
    Vidore3ComputerScienceRetrieval,
    Vidore3ComputerScienceRetrievalv2,
    Vidore3EnergyRetrieval,
    Vidore3EnergyRetrievalv2,
    Vidore3FinanceEnRetrieval,
    Vidore3FinanceEnRetrievalv2,
    Vidore3FinanceFrRetrieval,
    Vidore3FinanceFrRetrievalv2,
    Vidore3HrRetrieval,
    Vidore3HrRetrievalv2,
    Vidore3IndustrialRetrieval,
    Vidore3IndustrialRetrievalv2,
    Vidore3NuclearRetrieval,
    Vidore3NuclearRetrievalv2,
    Vidore3PharmaceuticalsRetrieval,
    Vidore3PharmaceuticalsRetrievalv2,
    Vidore3PhysicsRetrieval,
    Vidore3PhysicsRetrievalv2,
    Vidore3TelecomRetrieval,
    Vidore3TelecomRetrievalv2,
)
from .web_faq_retrieval import WebFAQRetrieval
from .wikipedia_retrieval_multilingual import WikipediaRetrievalMultilingual
from .wit_t2i_retrieval import WITI2TRetrieval, WITT2IRetrieval
from .x_flickr30k_co_t2i_retrieval import (
    XFlickr30kCoI2TRetrieval,
    XFlickr30kCoT2IRetrieval,
)
from .x_market_retrieval import XMarket
from .x_qu_ad_retrieval import XQuADRetrieval
from .xm3600_t2i_retrieval import XM3600I2TRetrieval, XM3600T2IRetrieval
from .xpqa_retrieval import XPQARetrieval

__all__ = [
    "AfriMCQAA2IRetrieval",
    "AfriMCQAI2ARetrieval",
    "AudioCapsA2TRetrieval",
    "AudioCapsT2ARetrieval",
    "BelebeleRetrieval",
    "CUREv1Retrieval",
    "CommonVoiceMini17A2TRetrieval",
    "CommonVoiceMini17T2ARetrieval",
    "CommonVoiceMini21A2TRetrieval",
    "CommonVoiceMini21T2ARetrieval",
    "CrossLingualSemanticDiscriminationWMT19",
    "CrossLingualSemanticDiscriminationWMT21",
    "EuroPIRQRetrieval",
    "FleursA2TRetrieval",
    "FleursA2TRetrievalV2",
    "FleursT2ARetrieval",
    "FleursT2ARetrievalV2",
    "GLAMI1MI2TRetrieval",
    "GLAMI1MT2IRetrieval",
    "GlobalNewsRetrieval",
    "GoogleSVQA2TRetrieval",
    "GoogleSVQT2ARetrieval",
    "IndicQARetrieval",
    "JamAltArtistA2ARetrieval",
    "JamAltLyricA2TRetrieval",
    "JamAltLyricT2ARetrieval",
    "JinaVDRAirbnbSyntheticRetrieval",
    "JinaVDRArabicChartQARetrieval",
    "JinaVDRArabicInfographicsVQARetrieval",
    "JinaVDRArxivQARetrieval",
    "JinaVDRAutomobileCatelogRetrieval",
    "JinaVDRBeveragesCatalogueRetrieval",
    "JinaVDRCharXivOCRRetrieval",
    "JinaVDRChartQARetrieval",
    "JinaVDRDocQAAI",
    "JinaVDRDocQAEnergyRetrieval",
    "JinaVDRDocQAGovReportRetrieval",
    "JinaVDRDocQAHealthcareIndustryRetrieval",
    "JinaVDRDocVQARetrieval",
    "JinaVDRDonutVQAISynHMPRetrieval",
    "JinaVDREuropeanaDeNewsRetrieval",
    "JinaVDREuropeanaEsNewsRetrieval",
    "JinaVDREuropeanaFrNewsRetrieval",
    "JinaVDREuropeanaItScansRetrieval",
    "JinaVDREuropeanaNlLegalRetrieval",
    "JinaVDRGitHubReadmeRetrieval",
    "JinaVDRHindiGovVQARetrieval",
    "JinaVDRHungarianDocQARetrieval",
    "JinaVDRInfovqaRetrieval",
    "JinaVDRJDocQARetrieval",
    "JinaVDRJina2024YearlyBookRetrieval",
    "JinaVDRMMTabRetrieval",
    "JinaVDRMPMQARetrieval",
    "JinaVDRMedicalPrescriptionsRetrieval",
    "JinaVDROWIDChartsRetrieval",
    "JinaVDROpenAINewsRetrieval",
    "JinaVDRPlotQARetrieval",
    "JinaVDRRamensBenchmarkRetrieval",
    "JinaVDRShanghaiMasterPlanRetrieval",
    "JinaVDRShiftProjectRetrieval",
    "JinaVDRStanfordSlideRetrieval",
    "JinaVDRStudentEnrollmentSyntheticRetrieval",
    "JinaVDRTQARetrieval",
    "JinaVDRTabFQuadRetrieval",
    "JinaVDRTableVQARetrieval",
    "JinaVDRTatQARetrieval",
    "JinaVDRTweetStockSyntheticsRetrieval",
    "JinaVDRWikimediaCommonsDocumentsRetrieval",
    "JinaVDRWikimediaCommonsMapsRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLRetrievalHardNegativesV2",
    "MIRACLVisionRetrieval",
    "MKQARetrieval",
    "MLQARetrieval",
    "MMarcoRetrievalMultilingual",
    "MintakaRetrieval",
    "MrTidyRetrieval",
    "MuPLeRRetrieval",
    "Multi30kI2TRetrieval",
    "Multi30kT2IRetrieval",
    "MultiLongDocRetrieval",
    "MultilingualNanoArguAnaRetrieval",
    "MultilingualNanoClimateFeverRetrieval",
    "MultilingualNanoDBPediaRetrieval",
    "MultilingualNanoFEVERRetrieval",
    "MultilingualNanoFiQA2018Retrieval",
    "MultilingualNanoHotpotQARetrieval",
    "MultilingualNanoMSMARCORetrieval",
    "MultilingualNanoNFCorpusRetrieval",
    "MultilingualNanoNQRetrieval",
    "MultilingualNanoQuoraRetrieval",
    "MultilingualNanoSCIDOCSRetrieval",
    "MultilingualNanoSciFactRetrieval",
    "MultilingualNanoTouche2020Retrieval",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2023RetrievalHardNegatives",
    "OmnilingualASRA2TRetrieval",
    "OmnilingualASRT2ARetrieval",
    "PublicHealthQARetrieval",
    "PublicNewsRetrieval",
    "RuSciBenchCiteRetrieval",
    "RuSciBenchCociteRetrieval",
    "SpokenWikipediaA2TRetrieval",
    "SpokenWikipediaT2ARetrieval",
    "StatcanDialogueDatasetRetrieval",
    "VDRMultilingualRetrieval",
    "VaaniA2TRetrieval",
    "VaaniT2ARetrieval",
    "Vidore2BioMedicalLecturesRetrieval",
    "Vidore2ESGReportsHLRetrieval",
    "Vidore2ESGReportsRetrieval",
    "Vidore2EconomicsReportsRetrieval",
    "Vidore3ComputerScienceRetrieval",
    "Vidore3ComputerScienceRetrievalv2",
    "Vidore3EnergyRetrieval",
    "Vidore3EnergyRetrievalv2",
    "Vidore3FinanceEnRetrieval",
    "Vidore3FinanceEnRetrievalv2",
    "Vidore3FinanceFrRetrieval",
    "Vidore3FinanceFrRetrievalv2",
    "Vidore3HrRetrieval",
    "Vidore3HrRetrievalv2",
    "Vidore3IndustrialRetrieval",
    "Vidore3IndustrialRetrievalv2",
    "Vidore3NuclearRetrieval",
    "Vidore3NuclearRetrievalv2",
    "Vidore3PharmaceuticalsRetrieval",
    "Vidore3PharmaceuticalsRetrievalv2",
    "Vidore3PhysicsRetrieval",
    "Vidore3PhysicsRetrievalv2",
    "Vidore3TelecomRetrieval",
    "Vidore3TelecomRetrievalv2",
    "WITI2TRetrieval",
    "WITT2IRetrieval",
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XFlickr30kCoI2TRetrieval",
    "XFlickr30kCoT2IRetrieval",
    "XM3600I2TRetrieval",
    "XM3600T2IRetrieval",
    "XMarket",
    "XPQARetrieval",
    "XQuADRetrieval",
]
