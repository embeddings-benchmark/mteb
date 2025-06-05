from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper


MODEL_PROMPTS = {
    "Classification": "Instruct: classify the query into different classes. \n Query: ",
    "MultilabelClassification": "Instruct: classify the query into different classes. \n Query: ",
    "Clustering": "Instruct: classify the query into different classes. \n Query: ",
    "Reranking-query": "Instruct: Given a query, retrieve documents that answer the query. \n Query: ",
    "Retrieval-query": "Instruct: Given a query, retrieve documents that answer the query. \n Query: ",
}

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
    "HotpotQAHardNegatives": ["train"],
    "MultiLongDocRetrieval": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCOv2": ["train"],
    "NFCorpus": ["train"],
    "SciFact": ["train"],
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
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

KaLM-Team__KaLM_Embedding_X_0605 = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name_or_path="KaLM-Team/KaLM-Embedding-X-0605",
        model_prompts=MODEL_PROMPTS,
        max_seq_length=512,
    ),
    name="KaLM-Team/KaLM-Embedding-X-0605",
    languages=None,
    open_weights=False,
    revision="1",
    release_date="2025-06-05",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=4096,
    license=None,
    reference="https://github.com/KaLM-Team/KaLM-Embedding-X",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/HITsz-TMG/KaLM-Embedding",
    public_training_data=None,
    training_datasets=kalm_training_data,
)
