from .belebele_retrieval import BelebeleRetrieval
from .cross_lingual_semantic_discrimination_wmt19 import (
    CrossLingualSemanticDiscriminationWMT19,
)
from .cross_lingual_semantic_discrimination_wmt21 import (
    CrossLingualSemanticDiscriminationWMT21,
)
from .cur_ev1_retrieval import CUREv1Retrieval
from .indic_qa_retrieval import IndicQARetrieval
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
from .mr_tidy_retrieval import MrTidyRetrieval
from .multi_long_doc_retrieval import MultiLongDocRetrieval
from .neu_clir2022_retrieval import (
    NeuCLIR2022Retrieval,
    NeuCLIR2022RetrievalHardNegatives,
)
from .neu_clir2023_retrieval import (
    NeuCLIR2023Retrieval,
    NeuCLIR2023RetrievalHardNegatives,
)
from .public_health_qa_retrieval import PublicHealthQARetrieval
from .ru_sci_bench_retrieval import RuSciBenchCiteRetrieval, RuSciBenchCociteRetrieval
from .statcan_dialogue_dataset_retrieval import StatcanDialogueDatasetRetrieval
from .vdr_multilingual_retrieval import VDRMultilingualRetrieval
from .vidore2_bench_retrieval import (
    Vidore2BioMedicalLecturesRetrieval,
    Vidore2EconomicsReportsRetrieval,
    Vidore2ESGReportsHLRetrieval,
    Vidore2ESGReportsRetrieval,
)
from .vidore3_bench_retrieval import (
    Vidore3ComputerScienceRetrieval,
    Vidore3EnergyRetrieval,
    Vidore3FinanceEnRetrieval,
    Vidore3FinanceFrRetrieval,
    Vidore3HrRetrieval,
    Vidore3IndustrialRetrieval,
    Vidore3NuclearRetrieval,
    Vidore3PharmaceuticalsRetrieval,
    Vidore3PhysicsRetrieval,
    Vidore3TelecomRetrieval,
)
from .web_faq_retrieval import WebFAQRetrieval
from .wikipedia_retrieval_multilingual import WikipediaRetrievalMultilingual
from .wit_t2i_retrieval import WITT2IRetrieval
from .x_flickr30k_co_t2i_retrieval import XFlickr30kCoT2IRetrieval
from .x_market_retrieval import XMarket
from .x_qu_ad_retrieval import XQuADRetrieval
from .xm3600_t2i_retrieval import XM3600T2IRetrieval
from .xpqa_retrieval import XPQARetrieval

__all__ = [
    "BelebeleRetrieval",
    "CUREv1Retrieval",
    "CrossLingualSemanticDiscriminationWMT19",
    "CrossLingualSemanticDiscriminationWMT21",
    "IndicQARetrieval",
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
    "MintakaRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2023RetrievalHardNegatives",
    "PublicHealthQARetrieval",
    "RuSciBenchCiteRetrieval",
    "RuSciBenchCociteRetrieval",
    "StatcanDialogueDatasetRetrieval",
    "VDRMultilingualRetrieval",
    "Vidore2BioMedicalLecturesRetrieval",
    "Vidore2ESGReportsHLRetrieval",
    "Vidore2ESGReportsRetrieval",
    "Vidore2EconomicsReportsRetrieval",
    "Vidore3ComputerScienceRetrieval",
    "Vidore3EnergyRetrieval",
    "Vidore3FinanceEnRetrieval",
    "Vidore3FinanceFrRetrieval",
    "Vidore3HrRetrieval",
    "Vidore3IndustrialRetrieval",
    "Vidore3NuclearRetrieval",
    "Vidore3PharmaceuticalsRetrieval",
    "Vidore3PhysicsRetrieval",
    "Vidore3TelecomRetrieval",
    "WITT2IRetrieval",
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XFlickr30kCoT2IRetrieval",
    "XM3600T2IRetrieval",
    "XMarket",
    "XPQARetrieval",
    "XQuADRetrieval",
]
