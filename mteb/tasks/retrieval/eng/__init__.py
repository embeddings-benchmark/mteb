from .aila_casedocs_retrieval import AILACasedocs
from .aila_statutes_retrieval import AILAStatutes
from .alpha_nli_retrieval import AlphaNLI
from .arc_challenge_retrieval import ARCChallenge
from .argu_ana_retrieval import ArguAna
from .bar_exam_qa_retrieval import BarExamQARetrieval
from .bill_sum_ca_retrieval import BillSumCARetrieval
from .bill_sum_us_retrieval import BillSumUSRetrieval
from .birco_argu_ana_reranking import BIRCOArguAnaReranking
from .birco_clinical_trial_reranking import BIRCOClinicalTrialReranking
from .birco_doris_mae_reranking import BIRCODorisMaeReranking
from .birco_relic_reranking import BIRCORelicReranking
from .birco_whats_that_book_reranking import BIRCOWhatsThatBookReranking
from .blink_it2i_retrieval import BLINKIT2IRetrieval
from .blink_it2t_retrieval import BLINKIT2TRetrieval
from .bright_retrieval import BrightLongRetrieval, BrightRetrieval
from .built_bench_retrieval import BuiltBenchRetrieval
from .chat_doctor_retrieval import ChatDoctorRetrieval
from .chem_hotpot_qa_retrieval import ChemHotpotQARetrieval
from .chem_nq_retrieval import ChemNQRetrieval
from .cirr_it2i_retrieval import CIRRIT2IRetrieval
from .climate_fever_retrieval import (
    ClimateFEVER,
    ClimateFEVERHardNegatives,
    ClimateFEVERHardNegativesV2,
    ClimateFEVERRetrievalv2,
)
from .cqa_dupstack_android_retrieval import CQADupstackAndroidRetrieval
from .cqa_dupstack_english_retrieval import CQADupstackEnglishRetrieval
from .cqa_dupstack_gaming_retrieval import CQADupstackGamingRetrieval
from .cqa_dupstack_gis_retrieval import CQADupstackGisRetrieval
from .cqa_dupstack_mathematica_retrieval import CQADupstackMathematicaRetrieval
from .cqa_dupstack_physics_retrieval import CQADupstackPhysicsRetrieval
from .cqa_dupstack_programmers_retrieval import CQADupstackProgrammersRetrieval
from .cqa_dupstack_stats_retrieval import CQADupstackStatsRetrieval
from .cqa_dupstack_tex_retrieval import CQADupstackTexRetrieval
from .cqa_dupstack_unix_retrieval import CQADupstackUnixRetrieval
from .cqa_dupstack_webmasters_retrieval import CQADupstackWebmastersRetrieval
from .cqa_dupstack_wordpress_retrieval import CQADupstackWordpressRetrieval
from .cub200_i2i_retrieval import CUB200I2I
from .dapfam_patent_retrieval import (
    DAPFAMAllTitlAbsClmToFullTextRetrieval,
    DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval,
    DAPFAMAllTitlAbsClmToTitlAbsRetrieval,
    DAPFAMAllTitlAbsToFullTextRetrieval,
    DAPFAMAllTitlAbsToTitlAbsClmRetrieval,
    DAPFAMAllTitlAbsToTitlAbsRetrieval,
    DAPFAMInTitlAbsClmToFullTextRetrieval,
    DAPFAMInTitlAbsClmToTitlAbsClmRetrieval,
    DAPFAMInTitlAbsClmToTitlAbsRetrieval,
    DAPFAMInTitlAbsToFullTextRetrieval,
    DAPFAMInTitlAbsToTitlAbsClmRetrieval,
    DAPFAMInTitlAbsToTitlAbsRetrieval,
    DAPFAMOutTitlAbsClmToFullTextRetrieval,
    DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval,
    DAPFAMOutTitlAbsClmToTitlAbsRetrieval,
    DAPFAMOutTitlAbsToFullTextRetrieval,
    DAPFAMOutTitlAbsToTitlAbsClmRetrieval,
    DAPFAMOutTitlAbsToTitlAbsRetrieval,
)
from .dbpedia_retrieval import DBPedia, DBPediaHardNegatives, DBPediaHardNegativesV2
from .edis_t2it_retrieval import EDIST2ITRetrieval
from .encyclopedia_vqa_it2it_retrieval import EncyclopediaVQAIT2ITRetrieval
from .english_finance1_retrieval import EnglishFinance1Retrieval
from .english_finance2_retrieval import EnglishFinance2Retrieval
from .english_finance3_retrieval import EnglishFinance3Retrieval
from .english_finance4_retrieval import EnglishFinance4Retrieval
from .english_healthcare1_retrieval import EnglishHealthcare1Retrieval
from .faith_dial_retrieval import FaithDialRetrieval
from .fashion200k_i2t_retrieval import Fashion200kI2TRetrieval
from .fashion200k_t2i_retrieval import Fashion200kT2IRetrieval
from .fashion_iq_it2i_retrieval import FashionIQIT2IRetrieval
from .feedback_qa_retrieval import FeedbackQARetrieval
from .fever_retrieval import FEVER, FEVERHardNegatives, FEVERHardNegativesV2
from .fi_qa2018_retrieval import FiQA2018
from .fin_qa_retrieval import FinQARetrieval
from .finance_bench_retrieval import FinanceBenchRetrieval
from .flickr30k_i2t_retrieval import Flickr30kI2TRetrieval
from .flickr30k_t2i_retrieval import Flickr30kT2IRetrieval
from .forb_i2i_retrieval import FORBI2I
from .gl_dv2_i2i_retrieval import GLDv2I2IRetrieval
from .gl_dv2_i2t_retrieval import GLDv2I2TRetrieval
from .gov_report_retrieval import GovReportRetrieval
from .hagrid_retrieval import HagridRetrieval
from .hateful_memes_i2t_retrieval import HatefulMemesI2TRetrieval
from .hateful_memes_t2i_retrieval import HatefulMemesT2IRetrieval
from .hc3_finance_retrieval import HC3FinanceRetrieval
from .hella_swag_retrieval import HellaSwag
from .hotpot_qa_retrieval import (
    HotpotQA,
    HotpotQAHardNegatives,
    HotpotQAHardNegativesV2,
)
from .image_co_de_t2i_retrieval import ImageCoDeT2IRetrieval
from .info_seek_it2it_retrieval import InfoSeekIT2ITRetrieval
from .info_seek_it2t_retrieval import InfoSeekIT2TRetrieval
from .legal_bench_consumer_contracts_qa_retrieval import LegalBenchConsumerContractsQA
from .legal_bench_corporate_lobbying_retrieval import LegalBenchCorporateLobbying
from .legal_summarization_retrieval import LegalSummarization
from .lemb_narrative_qa_retrieval import LEMBNarrativeQARetrieval
from .lemb_needle_retrieval import LEMBNeedleRetrieval
from .lemb_passkey_retrieval import LEMBPasskeyRetrieval
from .lemb_summ_screen_fd_retrieval import LEMBSummScreenFDRetrieval
from .lemb_wikim_qa_retrieval import LEMBWikimQARetrieval
from .lembqm_sum_retrieval import LEMBQMSumRetrieval
from .limit_retrieval import LIMITRetrieval, LIMITSmallRetrieval
from .lit_search_retrieval import LitSearchRetrieval
from .llava_it2t_retrieval import LLaVAIT2TRetrieval
from .lotte_retrieval import LoTTERetrieval
from .medical_qa_retrieval import MedicalQARetrieval
from .memotion_i2t_retrieval import MemotionI2TRetrieval
from .memotion_t2i_retrieval import MemotionT2IRetrieval
from .met_i2i_retrieval import METI2IRetrieval
from .ml_questions import MLQuestionsRetrieval
from .mscoco_i2t_retrieval import MSCOCOI2TRetrieval
from .mscoco_t2i_retrieval import MSCOCOT2IRetrieval
from .msmarc_ov2_retrieval import MSMARCOv2
from .msmarco_retrieval import MSMARCO, MSMARCOHardNegatives
from .nano_argu_ana_retrieval import NanoArguAnaRetrieval
from .nano_climate_fever_retrieval import NanoClimateFeverRetrieval
from .nano_db_pedia_retrieval import NanoDBPediaRetrieval
from .nano_fever_retrieval import NanoFEVERRetrieval
from .nano_fi_qa2018_retrieval import NanoFiQA2018Retrieval
from .nano_hotpot_qa_retrieval import NanoHotpotQARetrieval
from .nano_msmarco_retrieval import NanoMSMARCORetrieval
from .nano_nf_corpus_retrieval import NanoNFCorpusRetrieval
from .nano_nq_retrieval import NanoNQRetrieval
from .nano_quora_retrieval import NanoQuoraRetrieval
from .nano_sci_fact_retrieval import NanoSciFactRetrieval
from .nano_scidocs_retrieval import NanoSCIDOCSRetrieval
from .nano_touche2020_retrieval import NanoTouche2020Retrieval
from .narrative_qa_retrieval import NarrativeQARetrieval
from .nf_corpus_retrieval import NFCorpus
from .nights_i2i_retrieval import NIGHTSI2IRetrieval
from .nq_retrieval import NQ, NQHardNegatives
from .okvqa_it2t_retrieval import OKVQAIT2TRetrieval
from .oven_it2it_retrieval import OVENIT2ITRetrieval
from .oven_it2t_retrieval import OVENIT2TRetrieval
from .piqa_retrieval import PIQA
from .quail_retrieval import Quail
from .quora_retrieval import (
    QuoraRetrieval,
    QuoraRetrievalHardNegatives,
    QuoraRetrievalHardNegativesV2,
)
from .r2_med_retrieval import (
    R2MEDBioinformaticsRetrieval,
    R2MEDBiologyRetrieval,
    R2MEDIIYiClinicalRetrieval,
    R2MEDMedicalSciencesRetrieval,
    R2MEDMedQADiagRetrieval,
    R2MEDMedXpertQAExamRetrieval,
    R2MEDPMCClinicalRetrieval,
    R2MEDPMCTreatmentRetrieval,
)
from .r_oxford_i2i_retrieval import (
    ROxfordEasyI2IRetrieval,
    ROxfordHardI2IRetrieval,
    ROxfordMediumI2IRetrieval,
)
from .r_paris_i2i_retrieval import (
    RParisEasyI2IRetrieval,
    RParisHardI2IRetrieval,
    RParisMediumI2IRetrieval,
)
from .ra_rb_code_retrieval import RARbCode
from .ra_rb_math_retrieval import RARbMath
from .re_mu_q_it2t_retrieval import ReMuQIT2TRetrieval
from .rp2k_i2i_retrieval import RP2kI2IRetrieval
from .sci_fact_retrieval import SciFact
from .sci_mmir_i2t_retrieval import SciMMIRI2TRetrieval
from .sci_mmir_t2i_retrieval import SciMMIRT2IRetrieval
from .scidocs_retrieval import SCIDOCS
from .siqa_retrieval import SIQA
from .sketchy_i2i_retrieval import SketchyI2IRetrieval
from .sop_i2i_retrieval import SOPI2IRetrieval
from .spart_qa_retrieval import SpartQA
from .stanford_cars_i2i_retrieval import StanfordCarsI2I
from .temp_reason_l1_retrieval import TempReasonL1
from .temp_reason_l2_context_retrieval import TempReasonL2Context
from .temp_reason_l2_fact_retrieval import TempReasonL2Fact
from .temp_reason_l2_pure_retrieval import TempReasonL2Pure
from .temp_reason_l3_context_retrieval import TempReasonL3Context
from .temp_reason_l3_fact_retrieval import TempReasonL3Fact
from .temp_reason_l3_pure_retrieval import TempReasonL3Pure
from .topi_ocqa_retrieval import TopiOCQARetrieval, TopiOCQARetrievalHardNegatives
from .touche2020_retrieval import Touche2020, Touche2020v3Retrieval
from .treccovid_retrieval import TRECCOVID
from .trecdl_retrieval import TRECDL2019, TRECDL2020
from .tu_berlin_t2i_retrieval import TUBerlinT2IRetrieval
from .vidore_bench_retrieval import (
    VidoreArxivQARetrieval,
    VidoreDocVQARetrieval,
    VidoreInfoVQARetrieval,
    VidoreShiftProjectRetrieval,
    VidoreSyntheticDocQAAIRetrieval,
    VidoreSyntheticDocQAEnergyRetrieval,
    VidoreSyntheticDocQAGovernmentReportsRetrieval,
    VidoreSyntheticDocQAHealthcareIndustryRetrieval,
    VidoreTabfquadRetrieval,
    VidoreTatdqaRetrieval,
)
from .visual_news_i2t_retrieval import VisualNewsI2TRetrieval
from .visual_news_t2i_retrieval import VisualNewsT2IRetrieval
from .viz_wiz_it2t_retrieval import VizWizIT2TRetrieval
from .vqa2_it2t_retrieval import VQA2IT2TRetrieval
from .web_qa_t2it_retrieval import WebQAT2ITRetrieval
from .web_qa_t2t_retrieval import WebQAT2TRetrieval
from .wino_grande_retrieval import WinoGrande

__all__ = [
    "CUB200I2I",
    "FEVER",
    "FORBI2I",
    "MSMARCO",
    "NQ",
    "PIQA",
    "SCIDOCS",
    "SIQA",
    "TRECCOVID",
    "TRECDL2019",
    "TRECDL2020",
    "AILACasedocs",
    "AILAStatutes",
    "ARCChallenge",
    "AlphaNLI",
    "ArguAna",
    "BIRCOArguAnaReranking",
    "BIRCOClinicalTrialReranking",
    "BIRCODorisMaeReranking",
    "BIRCORelicReranking",
    "BIRCOWhatsThatBookReranking",
    "BLINKIT2IRetrieval",
    "BLINKIT2TRetrieval",
    "BarExamQARetrieval",
    "BillSumCARetrieval",
    "BillSumUSRetrieval",
    "BrightLongRetrieval",
    "BrightRetrieval",
    "BuiltBenchRetrieval",
    "CIRRIT2IRetrieval",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "ChatDoctorRetrieval",
    "ChemHotpotQARetrieval",
    "ChemNQRetrieval",
    "ClimateFEVER",
    "ClimateFEVERHardNegatives",
    "ClimateFEVERHardNegativesV2",
    "ClimateFEVERRetrievalv2",
    "DAPFAMAllTitlAbsClmToFullTextRetrieval",
    "DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval",
    "DAPFAMAllTitlAbsClmToTitlAbsRetrieval",
    "DAPFAMAllTitlAbsToFullTextRetrieval",
    "DAPFAMAllTitlAbsToTitlAbsClmRetrieval",
    "DAPFAMAllTitlAbsToTitlAbsRetrieval",
    "DAPFAMInTitlAbsClmToFullTextRetrieval",
    "DAPFAMInTitlAbsClmToTitlAbsClmRetrieval",
    "DAPFAMInTitlAbsClmToTitlAbsRetrieval",
    "DAPFAMInTitlAbsToFullTextRetrieval",
    "DAPFAMInTitlAbsToTitlAbsClmRetrieval",
    "DAPFAMInTitlAbsToTitlAbsRetrieval",
    "DAPFAMOutTitlAbsClmToFullTextRetrieval",
    "DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval",
    "DAPFAMOutTitlAbsClmToTitlAbsRetrieval",
    "DAPFAMOutTitlAbsToFullTextRetrieval",
    "DAPFAMOutTitlAbsToTitlAbsClmRetrieval",
    "DAPFAMOutTitlAbsToTitlAbsRetrieval",
    "DBPedia",
    "DBPediaHardNegatives",
    "DBPediaHardNegativesV2",
    "EDIST2ITRetrieval",
    "EncyclopediaVQAIT2ITRetrieval",
    "EnglishFinance1Retrieval",
    "EnglishFinance2Retrieval",
    "EnglishFinance3Retrieval",
    "EnglishFinance4Retrieval",
    "EnglishHealthcare1Retrieval",
    "FEVERHardNegatives",
    "FEVERHardNegativesV2",
    "FaithDialRetrieval",
    "Fashion200kI2TRetrieval",
    "Fashion200kT2IRetrieval",
    "FashionIQIT2IRetrieval",
    "FeedbackQARetrieval",
    "FiQA2018",
    "FinQARetrieval",
    "FinanceBenchRetrieval",
    "Flickr30kI2TRetrieval",
    "Flickr30kT2IRetrieval",
    "GLDv2I2IRetrieval",
    "GLDv2I2TRetrieval",
    "GovReportRetrieval",
    "HC3FinanceRetrieval",
    "HagridRetrieval",
    "HatefulMemesI2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "HellaSwag",
    "HotpotQA",
    "HotpotQAHardNegatives",
    "HotpotQAHardNegativesV2",
    "ImageCoDeT2IRetrieval",
    "InfoSeekIT2ITRetrieval",
    "InfoSeekIT2TRetrieval",
    "LEMBNarrativeQARetrieval",
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBQMSumRetrieval",
    "LEMBSummScreenFDRetrieval",
    "LEMBWikimQARetrieval",
    "LIMITRetrieval",
    "LIMITSmallRetrieval",
    "LLaVAIT2TRetrieval",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
    "LitSearchRetrieval",
    "LoTTERetrieval",
    "METI2IRetrieval",
    "MLQuestionsRetrieval",
    "MSCOCOI2TRetrieval",
    "MSCOCOT2IRetrieval",
    "MSMARCOHardNegatives",
    "MSMARCOv2",
    "MedicalQARetrieval",
    "MemotionI2TRetrieval",
    "MemotionT2IRetrieval",
    "NFCorpus",
    "NIGHTSI2IRetrieval",
    "NQHardNegatives",
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
    "NarrativeQARetrieval",
    "OKVQAIT2TRetrieval",
    "OVENIT2ITRetrieval",
    "OVENIT2TRetrieval",
    "Quail",
    "QuoraRetrieval",
    "QuoraRetrievalHardNegatives",
    "QuoraRetrievalHardNegativesV2",
    "R2MEDBioinformaticsRetrieval",
    "R2MEDBiologyRetrieval",
    "R2MEDIIYiClinicalRetrieval",
    "R2MEDMedQADiagRetrieval",
    "R2MEDMedXpertQAExamRetrieval",
    "R2MEDMedicalSciencesRetrieval",
    "R2MEDPMCClinicalRetrieval",
    "R2MEDPMCTreatmentRetrieval",
    "RARbCode",
    "RARbMath",
    "ROxfordEasyI2IRetrieval",
    "ROxfordHardI2IRetrieval",
    "ROxfordMediumI2IRetrieval",
    "RP2kI2IRetrieval",
    "RParisEasyI2IRetrieval",
    "RParisHardI2IRetrieval",
    "RParisMediumI2IRetrieval",
    "ReMuQIT2TRetrieval",
    "SOPI2IRetrieval",
    "SciFact",
    "SciMMIRI2TRetrieval",
    "SciMMIRT2IRetrieval",
    "SketchyI2IRetrieval",
    "SpartQA",
    "StanfordCarsI2I",
    "TUBerlinT2IRetrieval",
    "TempReasonL1",
    "TempReasonL2Context",
    "TempReasonL2Fact",
    "TempReasonL2Pure",
    "TempReasonL3Context",
    "TempReasonL3Fact",
    "TempReasonL3Pure",
    "TopiOCQARetrieval",
    "TopiOCQARetrievalHardNegatives",
    "Touche2020",
    "Touche2020v3Retrieval",
    "VQA2IT2TRetrieval",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VisualNewsI2TRetrieval",
    "VisualNewsT2IRetrieval",
    "VizWizIT2TRetrieval",
    "WebQAT2ITRetrieval",
    "WebQAT2TRetrieval",
    "WinoGrande",
]
