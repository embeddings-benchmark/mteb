from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

kalm_training_data = {
    # from technical report
    # not in MTEB:
    # ExpertQA
    # MEDI2BGE
    # OpenOrca
    # PAQ
    # PubMedQA
    # SearchQA
    # arxiv_qa
    # rag-dataset-12000
    # CC-News
    # SQuAD 2.0
    # TriviaQA
    # WebGPT Comparisons
    # MultiNLI
    # NLLB
    # WikiAnswers
    # SimCSE NLI
    # SNLI
    # Aya Dataset
    # eli5
    # ----
    # in MTEB:
    "CodeFeedbackMT": ["train"],
    "CodeFeedbackST": ["train"],
    "ArxivClusteringP2P": ["train"],
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringP2P.v2": ["train"],
    "TRECCOVID": ["train"],
    "DBPedia": ["train"],
    "ESCIReranking": ["train"],
    "FEVER": ["train"],
    "FiQA2018": ["train"],
    "FEVERHardNegatives": ["train"],
    "NanoFEVERRetrieval": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "FiQA2018-NL": ["train"],  # translation not trained on
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "MultiLongDocRetrieval": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "MSMARCOv2": ["train"],
    "NFCorpus": ["train"],
    "SciFact": ["train"],
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
    "YahooAnswersTopicsClassification": ["train"],
    "ContractNLIConfidentialityOfAgreementLegalBenchClassification": ["train"],
    "ContractNLIExplicitIdentificationLegalBenchClassification": ["train"],
    "ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLILimitedUseLegalBenchClassification": ["train"],
    "ContractNLINoLicensingLegalBenchClassification": ["train"],
    "ContractNLINoticeOnCompelledDisclosureLegalBenchClassification": ["train"],
    "ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLIPermissibleCopyLegalBenchClassification": ["train"],
    "ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification": ["train"],
    "ContractNLIReturnOfConfidentialInformationLegalBenchClassification": ["train"],
    "ContractNLISharingWithEmployeesLegalBenchClassification": ["train"],
    "ContractNLISharingWithThirdPartiesLegalBenchClassification": ["train"],
    "ContractNLISurvivalOfObligationsLegalBenchClassification": ["train"],
    "QuoraRetrieval": ["train"],
    "NanoQuoraRetrieval": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "Banking77Classification": ["train"],
    "AmazonPolarityClassification": ["train"],
    "ImdbClassification": ["train"],
    "EmotionClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "MrTidyRetrieval": ["train"],
    "PawsXPairClassification": ["train"],
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "MultilingualSentiment": ["train"],
    "MassiveIntentClassification": ["train"],
    "MassiveScenarioClassification": ["train"],
    "MTOPDomainClassification": ["train"],
    "MTOPIntentClassification": ["train"],
}

HIT_TMG_INSTRUCTION = "Instruct: {instruction}\nQuery: "

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name_or_path="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=True,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    revision="45e42c89990c40aca042659133fc8b13c28634b5",
    release_date="2024-10-23",
    languages=["eng", "zho"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=kalm_training_data,  # Replace with actual dataset if available
    adapted_from="Qwen/Qwen2-0.5B",
    superseded_by=None,
)

HIT_TMG__KaLM_embedding_multilingual_mini_v1 = ModelMeta(
    name="HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    revision="8a82a0cd2b322b91723e252486f7cce6fd8ac9d3",
    release_date="2024-08-27",
    languages=["eng", "zho"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    similarity_fn_name="cosine",
    use_instructions=None,
    training_datasets=kalm_training_data,
    adapted_from="Qwen/Qwen2-0.5B",
    superseded_by=None,
)

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name_or_path="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=True,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
    release_date="2024-12-26",
    languages=["eng", "zho"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=kalm_training_data,
    adapted_from="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    superseded_by=None,
)
