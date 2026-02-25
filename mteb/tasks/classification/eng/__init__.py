from mteb.tasks.classification.eng.common_language_age_detection import (
    CommonLanguageAgeDetection,
)
from mteb.tasks.classification.eng.common_language_gender_detection import (
    CommonLanguageGenderDetection,
)
from mteb.tasks.classification.eng.common_language_language_classification import (
    CommonLanguageLanguageClassification,
)
from mteb.tasks.classification.eng.cremad import CREMAD
from mteb.tasks.classification.eng.cstr_vctk_accent_id import CSTRVCTKAccentID
from mteb.tasks.classification.eng.cstr_vctk_gender_classification import (
    CSTRVCTKGenderClassification,
)
from mteb.tasks.classification.eng.expresso import (
    ExpressoConvEmotionClassification,
    ExpressoReadEmotionClassification,
)
from mteb.tasks.classification.eng.fsdd import FSDD
from mteb.tasks.classification.eng.globe_v2_age_classification import (
    GlobeV2AgeClassification,
)
from mteb.tasks.classification.eng.globe_v2_gender_classification import (
    GlobeV2GenderClassification,
)
from mteb.tasks.classification.eng.globe_v3_age_classification import (
    GlobeV3AgeClassification,
)
from mteb.tasks.classification.eng.globe_v3_gender_classification import (
    GlobeV3GenderClassification,
)
from mteb.tasks.classification.eng.iemocap_emotion import IEMOCAPEmotionClassification
from mteb.tasks.classification.eng.iemocap_gender import IEMOCAPGenderClassification
from mteb.tasks.classification.eng.libri_count import LibriCount
from mteb.tasks.classification.eng.speech_commands import SpeechCommandsClassification
from mteb.tasks.classification.eng.spoke_n import SpokeNEnglishClassification
from mteb.tasks.classification.eng.spoken_q_afor_ic import SpokenQAForIC
from mteb.tasks.classification.eng.vocal_sound import VocalSoundClassification
from mteb.tasks.classification.eng.vox_celeb_sa import VoxCelebSA
from mteb.tasks.classification.eng.vox_lingua107_top10 import VoxLingua107Top10
from mteb.tasks.classification.eng.vox_populi_accent_id import VoxPopuliAccentID

from .amazon_polarity_classification import (
    AmazonPolarityClassification,
    AmazonPolarityClassificationV2,
)
from .arxiv_classification import ArxivClassification, ArxivClassificationV2
from .banking77_classification import Banking77Classification, Banking77ClassificationV2
from .birdsnap_classification import BirdsnapClassification
from .caltech101_classification import Caltech101Classification
from .cifar import CIFAR10Classification, CIFAR100Classification
from .country211_classification import Country211Classification
from .dbpedia_classification import DBpediaClassification, DBpediaClassificationV2
from .dtd_classification import DTDClassification
from .emotion_classification import EmotionClassification, EmotionClassificationV2
from .euro_sat_classification import EuroSATClassification
from .fer2013_classification import FER2013Classification
from .fgvc_aircraft_classification import FGVCAircraftClassification
from .financial_phrasebank_classification import (
    FinancialPhrasebankClassification,
    FinancialPhrasebankClassificationV2,
)
from .food101_classification import Food101Classification
from .frenk_en_classification import FrenkEnClassification, FrenkEnClassificationV2
from .gtsrb_classification import GTSRBClassification
from .hume_emotion_classification import HUMEEmotionClassification
from .hume_toxic_conversations_classification import (
    HUMEToxicConversationsClassification,
)
from .hume_tweet_sentiment_extraction_classification import (
    HUMETweetSentimentExtractionClassification,
)
from .imagenet1k import Imagenet1kClassification
from .imdb_classification import ImdbClassification, ImdbClassificationV2
from .legal_bench_classification import (
    CanadaTaxCourtOutcomesLegalBenchClassification,
    ContractNLIConfidentialityOfAgreementLegalBenchClassification,
    ContractNLIExplicitIdentificationLegalBenchClassification,
    ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification,
    ContractNLILimitedUseLegalBenchClassification,
    ContractNLINoLicensingLegalBenchClassification,
    ContractNLINoticeOnCompelledDisclosureLegalBenchClassification,
    ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification,
    ContractNLIPermissibleCopyLegalBenchClassification,
    ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification,
    ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification,
    ContractNLIReturnOfConfidentialInformationLegalBenchClassification,
    ContractNLISharingWithEmployeesLegalBenchClassification,
    ContractNLISharingWithThirdPartiesLegalBenchClassification,
    ContractNLISurvivalOfObligationsLegalBenchClassification,
    CorporateLobbyingLegalBenchClassification,
    CUADAffiliateLicenseLicenseeLegalBenchClassification,
    CUADAffiliateLicenseLicensorLegalBenchClassification,
    CUADAntiAssignmentLegalBenchClassification,
    CUADAuditRightsLegalBenchClassification,
    CUADCapOnLiabilityLegalBenchClassification,
    CUADChangeOfControlLegalBenchClassification,
    CUADCompetitiveRestrictionExceptionLegalBenchClassification,
    CUADCovenantNotToSueLegalBenchClassification,
    CUADEffectiveDateLegalBenchClassification,
    CUADExclusivityLegalBenchClassification,
    CUADExpirationDateLegalBenchClassification,
    CUADGoverningLawLegalBenchClassification,
    CUADInsuranceLegalBenchClassification,
    CUADIPOwnershipAssignmentLegalBenchClassification,
    CUADIrrevocableOrPerpetualLicenseLegalBenchClassification,
    CUADJointIPOwnershipLegalBenchClassification,
    CUADLicenseGrantLegalBenchClassification,
    CUADLiquidatedDamagesLegalBenchClassification,
    CUADMinimumCommitmentLegalBenchClassification,
    CUADMostFavoredNationLegalBenchClassification,
    CUADNonCompeteLegalBenchClassification,
    CUADNonDisparagementLegalBenchClassification,
    CUADNonTransferableLicenseLegalBenchClassification,
    CUADNoSolicitOfCustomersLegalBenchClassification,
    CUADNoSolicitOfEmployeesLegalBenchClassification,
    CUADNoticePeriodToTerminateRenewalLegalBenchClassification,
    CUADPostTerminationServicesLegalBenchClassification,
    CUADPriceRestrictionsLegalBenchClassification,
    CUADRenewalTermLegalBenchClassification,
    CUADRevenueProfitSharingLegalBenchClassification,
    CUADRofrRofoRofnLegalBenchClassification,
    CUADSourceCodeEscrowLegalBenchClassification,
    CUADTerminationForConvenienceLegalBenchClassification,
    CUADThirdPartyBeneficiaryLegalBenchClassification,
    CUADUncappedLiabilityLegalBenchClassification,
    CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification,
    CUADVolumeRestrictionLegalBenchClassification,
    CUADWarrantyDurationLegalBenchClassification,
    DefinitionClassificationLegalBenchClassification,
    Diversity1LegalBenchClassification,
    Diversity2LegalBenchClassification,
    Diversity3LegalBenchClassification,
    Diversity4LegalBenchClassification,
    Diversity5LegalBenchClassification,
    Diversity6LegalBenchClassification,
    FunctionOfDecisionSectionLegalBenchClassification,
    InsurancePolicyInterpretationLegalBenchClassification,
    InternationalCitizenshipQuestionsLegalBenchClassification,
    JCrewBlockerLegalBenchClassification,
    JCrewBlockerLegalBenchClassificationV2,
    LearnedHandsBenefitsLegalBenchClassification,
    LearnedHandsBusinessLegalBenchClassification,
    LearnedHandsConsumerLegalBenchClassification,
    LearnedHandsCourtsLegalBenchClassification,
    LearnedHandsCrimeLegalBenchClassification,
    LearnedHandsDivorceLegalBenchClassification,
    LearnedHandsDomesticViolenceLegalBenchClassification,
    LearnedHandsEducationLegalBenchClassification,
    LearnedHandsEmploymentLegalBenchClassification,
    LearnedHandsEstatesLegalBenchClassification,
    LearnedHandsFamilyLegalBenchClassification,
    LearnedHandsHealthLegalBenchClassification,
    LearnedHandsHousingLegalBenchClassification,
    LearnedHandsImmigrationLegalBenchClassification,
    LearnedHandsTortsLegalBenchClassification,
    LearnedHandsTrafficLegalBenchClassification,
    LegalReasoningCausalityLegalBenchClassification,
    LegalReasoningCausalityLegalBenchClassificationV2,
    MAUDLegalBenchClassification,
    MAUDLegalBenchClassificationV2,
    NYSJudicialEthicsLegalBenchClassification,
    OPP115DataRetentionLegalBenchClassification,
    OPP115DataSecurityLegalBenchClassification,
    OPP115DataSecurityLegalBenchClassificationV2,
    OPP115DoNotTrackLegalBenchClassification,
    OPP115DoNotTrackLegalBenchClassificationV2,
    OPP115FirstPartyCollectionUseLegalBenchClassification,
    OPP115InternationalAndSpecificAudiencesLegalBenchClassification,
    OPP115PolicyChangeLegalBenchClassification,
    OPP115ThirdPartySharingCollectionLegalBenchClassification,
    OPP115UserAccessEditAndDeletionLegalBenchClassification,
    OPP115UserChoiceControlLegalBenchClassification,
    OPP115UserChoiceControlLegalBenchClassificationV2,
    OralArgumentQuestionPurposeLegalBenchClassification,
    OralArgumentQuestionPurposeLegalBenchClassificationV2,
    OverrulingLegalBenchClassification,
    OverrulingLegalBenchClassificationV2,
    PersonalJurisdictionLegalBenchClassification,
    PROALegalBenchClassification,
    SCDBPAccountabilityLegalBenchClassification,
    SCDBPAuditsLegalBenchClassification,
    SCDBPCertificationLegalBenchClassification,
    SCDBPTrainingLegalBenchClassification,
    SCDBPVerificationLegalBenchClassification,
    SCDDAccountabilityLegalBenchClassification,
    SCDDAuditsLegalBenchClassification,
    SCDDCertificationLegalBenchClassification,
    SCDDTrainingLegalBenchClassification,
    SCDDVerificationLegalBenchClassification,
    TelemarketingSalesRuleLegalBenchClassification,
    TextualismToolDictionariesLegalBenchClassification,
    TextualismToolPlainLegalBenchClassification,
    UCCVCommonLawLegalBenchClassification,
    UnfairTOSLegalBenchClassification,
)
from .mnist_classification import MNISTClassification
from .news_classification import NewsClassification, NewsClassificationV2
from .oxford_flowers_classification import OxfordFlowersClassification
from .oxford_pets_classification import OxfordPetsClassification
from .patch_camelyon_classification import PatchCamelyonClassification
from .patent_classification import PatentClassification, PatentClassificationV2
from .poem_sentiment_classification import (
    PoemSentimentClassification,
    PoemSentimentClassificationV2,
)
from .resisc45_classification import RESISC45Classification
from .sds_eye_protection_classification import (
    SDSEyeProtectionClassification,
    SDSEyeProtectionClassificationV2,
)
from .sds_gloves_classification import (
    SDSGlovesClassification,
    SDSGlovesClassificationV2,
)
from .stanford_cars_classification import StanfordCarsClassification
from .stl10_classification import STL10Classification
from .sun397_classification import SUN397Classification
from .toxic_chat_classification import (
    ToxicChatClassification,
    ToxicChatClassificationV2,
)
from .toxic_conversations_classification import (
    ToxicConversationsClassification,
    ToxicConversationsClassificationV2,
)
from .tweet_sentiment_extraction_classification import (
    TweetSentimentExtractionClassification,
    TweetSentimentExtractionClassificationV2,
)
from .tweet_topic_single_classification import (
    TweetTopicSingleClassification,
    TweetTopicSingleClassificationV2,
)
from .ucf101_classification import UCF101Classification
from .wikipedia_bio_met_chem_classification import (
    WikipediaBioMetChemClassification,
    WikipediaBioMetChemClassificationV2,
)
from .wikipedia_biolum_neurochem_classification import (
    WikipediaBiolumNeurochemClassification,
)
from .wikipedia_chem_eng_specialties_classification import (
    WikipediaChemEngSpecialtiesClassification,
)
from .wikipedia_chem_fields_classification import (
    WikipediaChemFieldsClassification,
    WikipediaChemFieldsClassificationV2,
)
from .wikipedia_chemistry_topics_classification import (
    WikipediaChemistryTopicsClassification,
)
from .wikipedia_comp_chem_spectroscopy_classification import (
    WikipediaCompChemSpectroscopyClassification,
    WikipediaCompChemSpectroscopyClassificationV2,
)
from .wikipedia_cryobiology_separation_classification import (
    WikipediaCryobiologySeparationClassification,
)
from .wikipedia_crystallography_analytical_classification import (
    WikipediaCrystallographyAnalyticalClassification,
    WikipediaCrystallographyAnalyticalClassificationV2,
)
from .wikipedia_greenhouse_enantiopure_classification import (
    WikipediaGreenhouseEnantiopureClassification,
)
from .wikipedia_isotopes_fission_classification import (
    WikipediaIsotopesFissionClassification,
)
from .wikipedia_luminescence_classification import WikipediaLuminescenceClassification
from .wikipedia_organic_inorganic_classification import (
    WikipediaOrganicInorganicClassification,
)
from .wikipedia_salts_semiconductors_classification import (
    WikipediaSaltsSemiconductorsClassification,
)
from .wikipedia_solid_state_colloidal_classification import (
    WikipediaSolidStateColloidalClassification,
)
from .wikipedia_theoretical_applied_classification import (
    WikipediaTheoreticalAppliedClassification,
    WikipediaTheoreticalAppliedClassificationV2,
)
from .yahoo_answers_topics_classification import (
    YahooAnswersTopicsClassification,
    YahooAnswersTopicsClassificationV2,
)
from .yelp_review_full_classification import (
    YelpReviewFullClassification,
    YelpReviewFullClassificationV2,
)

__all__ = [
    "CREMAD",
    "FSDD",
    "AmazonPolarityClassification",
    "AmazonPolarityClassificationV2",
    "ArxivClassification",
    "ArxivClassificationV2",
    "Banking77Classification",
    "Banking77ClassificationV2",
    "BirdsnapClassification",
    "CIFAR10Classification",
    "CIFAR100Classification",
    "CSTRVCTKAccentID",
    "CSTRVCTKGenderClassification",
    "CUADAffiliateLicenseLicenseeLegalBenchClassification",
    "CUADAffiliateLicenseLicensorLegalBenchClassification",
    "CUADAntiAssignmentLegalBenchClassification",
    "CUADAuditRightsLegalBenchClassification",
    "CUADCapOnLiabilityLegalBenchClassification",
    "CUADChangeOfControlLegalBenchClassification",
    "CUADCompetitiveRestrictionExceptionLegalBenchClassification",
    "CUADCovenantNotToSueLegalBenchClassification",
    "CUADEffectiveDateLegalBenchClassification",
    "CUADExclusivityLegalBenchClassification",
    "CUADExpirationDateLegalBenchClassification",
    "CUADGoverningLawLegalBenchClassification",
    "CUADIPOwnershipAssignmentLegalBenchClassification",
    "CUADInsuranceLegalBenchClassification",
    "CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
    "CUADJointIPOwnershipLegalBenchClassification",
    "CUADLicenseGrantLegalBenchClassification",
    "CUADLiquidatedDamagesLegalBenchClassification",
    "CUADMinimumCommitmentLegalBenchClassification",
    "CUADMostFavoredNationLegalBenchClassification",
    "CUADNoSolicitOfCustomersLegalBenchClassification",
    "CUADNoSolicitOfEmployeesLegalBenchClassification",
    "CUADNonCompeteLegalBenchClassification",
    "CUADNonDisparagementLegalBenchClassification",
    "CUADNonTransferableLicenseLegalBenchClassification",
    "CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
    "CUADPostTerminationServicesLegalBenchClassification",
    "CUADPriceRestrictionsLegalBenchClassification",
    "CUADRenewalTermLegalBenchClassification",
    "CUADRevenueProfitSharingLegalBenchClassification",
    "CUADRofrRofoRofnLegalBenchClassification",
    "CUADSourceCodeEscrowLegalBenchClassification",
    "CUADTerminationForConvenienceLegalBenchClassification",
    "CUADThirdPartyBeneficiaryLegalBenchClassification",
    "CUADUncappedLiabilityLegalBenchClassification",
    "CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
    "CUADVolumeRestrictionLegalBenchClassification",
    "CUADWarrantyDurationLegalBenchClassification",
    "Caltech101Classification",
    "CanadaTaxCourtOutcomesLegalBenchClassification",
    "CommonLanguageAgeDetection",
    "CommonLanguageGenderDetection",
    "CommonLanguageLanguageClassification",
    "ContractNLIConfidentialityOfAgreementLegalBenchClassification",
    "ContractNLIExplicitIdentificationLegalBenchClassification",
    "ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
    "ContractNLILimitedUseLegalBenchClassification",
    "ContractNLINoLicensingLegalBenchClassification",
    "ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
    "ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
    "ContractNLIPermissibleCopyLegalBenchClassification",
    "ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
    "ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
    "ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
    "ContractNLISharingWithEmployeesLegalBenchClassification",
    "ContractNLISharingWithThirdPartiesLegalBenchClassification",
    "ContractNLISurvivalOfObligationsLegalBenchClassification",
    "CorporateLobbyingLegalBenchClassification",
    "Country211Classification",
    "DBpediaClassification",
    "DBpediaClassificationV2",
    "DTDClassification",
    "DefinitionClassificationLegalBenchClassification",
    "Diversity1LegalBenchClassification",
    "Diversity2LegalBenchClassification",
    "Diversity3LegalBenchClassification",
    "Diversity4LegalBenchClassification",
    "Diversity5LegalBenchClassification",
    "Diversity6LegalBenchClassification",
    "EmotionClassification",
    "EmotionClassificationV2",
    "EuroSATClassification",
    "ExpressoConvEmotionClassification",
    "ExpressoReadEmotionClassification",
    "FER2013Classification",
    "FGVCAircraftClassification",
    "FinancialPhrasebankClassification",
    "FinancialPhrasebankClassificationV2",
    "Food101Classification",
    "FrenkEnClassification",
    "FrenkEnClassificationV2",
    "FunctionOfDecisionSectionLegalBenchClassification",
    "GTSRBClassification",
    "GlobeV2AgeClassification",
    "GlobeV2GenderClassification",
    "GlobeV3AgeClassification",
    "GlobeV3GenderClassification",
    "HUMEEmotionClassification",
    "HUMEToxicConversationsClassification",
    "HUMETweetSentimentExtractionClassification",
    "IEMOCAPEmotionClassification",
    "IEMOCAPGenderClassification",
    "Imagenet1kClassification",
    "ImdbClassification",
    "ImdbClassificationV2",
    "InsurancePolicyInterpretationLegalBenchClassification",
    "InternationalCitizenshipQuestionsLegalBenchClassification",
    "JCrewBlockerLegalBenchClassification",
    "JCrewBlockerLegalBenchClassificationV2",
    "LearnedHandsBenefitsLegalBenchClassification",
    "LearnedHandsBusinessLegalBenchClassification",
    "LearnedHandsConsumerLegalBenchClassification",
    "LearnedHandsCourtsLegalBenchClassification",
    "LearnedHandsCrimeLegalBenchClassification",
    "LearnedHandsDivorceLegalBenchClassification",
    "LearnedHandsDomesticViolenceLegalBenchClassification",
    "LearnedHandsEducationLegalBenchClassification",
    "LearnedHandsEmploymentLegalBenchClassification",
    "LearnedHandsEstatesLegalBenchClassification",
    "LearnedHandsFamilyLegalBenchClassification",
    "LearnedHandsHealthLegalBenchClassification",
    "LearnedHandsHousingLegalBenchClassification",
    "LearnedHandsImmigrationLegalBenchClassification",
    "LearnedHandsTortsLegalBenchClassification",
    "LearnedHandsTrafficLegalBenchClassification",
    "LegalReasoningCausalityLegalBenchClassification",
    "LegalReasoningCausalityLegalBenchClassificationV2",
    "LibriCount",
    "MAUDLegalBenchClassification",
    "MAUDLegalBenchClassificationV2",
    "MNISTClassification",
    "NYSJudicialEthicsLegalBenchClassification",
    "NewsClassification",
    "NewsClassificationV2",
    "OPP115DataRetentionLegalBenchClassification",
    "OPP115DataSecurityLegalBenchClassification",
    "OPP115DataSecurityLegalBenchClassificationV2",
    "OPP115DoNotTrackLegalBenchClassification",
    "OPP115DoNotTrackLegalBenchClassificationV2",
    "OPP115FirstPartyCollectionUseLegalBenchClassification",
    "OPP115InternationalAndSpecificAudiencesLegalBenchClassification",
    "OPP115PolicyChangeLegalBenchClassification",
    "OPP115ThirdPartySharingCollectionLegalBenchClassification",
    "OPP115UserAccessEditAndDeletionLegalBenchClassification",
    "OPP115UserChoiceControlLegalBenchClassification",
    "OPP115UserChoiceControlLegalBenchClassificationV2",
    "OralArgumentQuestionPurposeLegalBenchClassification",
    "OralArgumentQuestionPurposeLegalBenchClassificationV2",
    "OverrulingLegalBenchClassification",
    "OverrulingLegalBenchClassificationV2",
    "OxfordFlowersClassification",
    "OxfordPetsClassification",
    "PROALegalBenchClassification",
    "PatchCamelyonClassification",
    "PatentClassification",
    "PatentClassificationV2",
    "PersonalJurisdictionLegalBenchClassification",
    "PoemSentimentClassification",
    "PoemSentimentClassificationV2",
    "RESISC45Classification",
    "SCDBPAccountabilityLegalBenchClassification",
    "SCDBPAuditsLegalBenchClassification",
    "SCDBPCertificationLegalBenchClassification",
    "SCDBPTrainingLegalBenchClassification",
    "SCDBPVerificationLegalBenchClassification",
    "SCDDAccountabilityLegalBenchClassification",
    "SCDDAuditsLegalBenchClassification",
    "SCDDCertificationLegalBenchClassification",
    "SCDDTrainingLegalBenchClassification",
    "SCDDVerificationLegalBenchClassification",
    "SDSEyeProtectionClassification",
    "SDSEyeProtectionClassificationV2",
    "SDSGlovesClassification",
    "SDSGlovesClassificationV2",
    "STL10Classification",
    "SUN397Classification",
    "SpeechCommandsClassification",
    "SpokeNEnglishClassification",
    "SpokenQAForIC",
    "StanfordCarsClassification",
    "TelemarketingSalesRuleLegalBenchClassification",
    "TextualismToolDictionariesLegalBenchClassification",
    "TextualismToolPlainLegalBenchClassification",
    "ToxicChatClassification",
    "ToxicChatClassificationV2",
    "ToxicConversationsClassification",
    "ToxicConversationsClassificationV2",
    "TweetSentimentExtractionClassification",
    "TweetSentimentExtractionClassificationV2",
    "TweetTopicSingleClassification",
    "TweetTopicSingleClassificationV2",
    "UCCVCommonLawLegalBenchClassification",
    "UCF101Classification",
    "UnfairTOSLegalBenchClassification",
    "VocalSoundClassification",
    "VoxCelebSA",
    "VoxLingua107Top10",
    "VoxPopuliAccentID",
    "WikipediaBioMetChemClassification",
    "WikipediaBioMetChemClassificationV2",
    "WikipediaBiolumNeurochemClassification",
    "WikipediaChemEngSpecialtiesClassification",
    "WikipediaChemFieldsClassification",
    "WikipediaChemFieldsClassificationV2",
    "WikipediaChemistryTopicsClassification",
    "WikipediaCompChemSpectroscopyClassification",
    "WikipediaCompChemSpectroscopyClassificationV2",
    "WikipediaCryobiologySeparationClassification",
    "WikipediaCrystallographyAnalyticalClassification",
    "WikipediaCrystallographyAnalyticalClassificationV2",
    "WikipediaGreenhouseEnantiopureClassification",
    "WikipediaIsotopesFissionClassification",
    "WikipediaLuminescenceClassification",
    "WikipediaOrganicInorganicClassification",
    "WikipediaSaltsSemiconductorsClassification",
    "WikipediaSolidStateColloidalClassification",
    "WikipediaTheoreticalAppliedClassification",
    "WikipediaTheoreticalAppliedClassificationV2",
    "YahooAnswersTopicsClassification",
    "YahooAnswersTopicsClassificationV2",
    "YelpReviewFullClassification",
    "YelpReviewFullClassificationV2",
]
