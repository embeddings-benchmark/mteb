from .BelebeleRetrieval import BelebeleRetrieval
from .CrossLingualSemanticDiscriminationWMT19 import (
    CrossLingualSemanticDiscriminationWMT19,
)
from .CrossLingualSemanticDiscriminationWMT21 import (
    CrossLingualSemanticDiscriminationWMT21,
)
from .CUREv1Retrieval import CUREv1Retrieval
from .IndicQARetrieval import IndicQARetrieval
from .JinaVDRBenchRetrieval import (
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
from .MintakaRetrieval import MintakaRetrieval
from .MIRACLRetrieval import (
    MIRACLRetrieval,
    MIRACLRetrievalHardNegatives,
    MIRACLRetrievalHardNegativesV2,
)
from .MIRACLVisionRetrieval import MIRACLVisionRetrieval
from .MKQARetrieval import MKQARetrieval
from .MLQARetrieval import MLQARetrieval
from .MrTidyRetrieval import MrTidyRetrieval
from .MultiLongDocRetrieval import MultiLongDocRetrieval
from .NeuCLIR2022Retrieval import (
    NeuCLIR2022Retrieval,
    NeuCLIR2022RetrievalHardNegatives,
)
from .NeuCLIR2023Retrieval import (
    NeuCLIR2023Retrieval,
    NeuCLIR2023RetrievalHardNegatives,
)
from .PublicHealthQARetrieval import PublicHealthQARetrieval
from .RuSciBenchRetrieval import RuSciBenchCiteRetrieval, RuSciBenchCociteRetrieval
from .StatcanDialogueDatasetRetrieval import StatcanDialogueDatasetRetrieval
from .VdrMultilingualRetrieval import VDRMultilingualRetrieval
from .Vidore2BenchRetrieval import (
    Vidore2BioMedicalLecturesRetrieval,
    Vidore2EconomicsReportsRetrieval,
    Vidore2ESGReportsHLRetrieval,
    Vidore2ESGReportsRetrieval,
)
from .WebFAQRetrieval import WebFAQRetrieval
from .WikipediaRetrievalMultilingual import WikipediaRetrievalMultilingual
from .WITT2IRetrieval import WITT2IRetrieval
from .XFlickr30kCoT2IRetrieval import XFlickr30kCoT2IRetrieval
from .XM3600T2IRetrieval import XM3600T2IRetrieval
from .XMarketRetrieval import XMarket
from .XPQARetrieval import XPQARetrieval
from .XQuADRetrieval import XQuADRetrieval

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
    "WITT2IRetrieval",
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XFlickr30kCoT2IRetrieval",
    "XM3600T2IRetrieval",
    "XMarket",
    "XPQARetrieval",
    "XQuADRetrieval",
]
