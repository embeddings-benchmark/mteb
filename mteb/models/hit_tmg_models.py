from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)


class KALMWrapper(InstructSentenceTransformerWrapper):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.add_eos_token:
            sentences = [
                example + self.model.tokenizer.eos_token for example in sentences
            ]

        instruction = self.get_task_instruction(
            task_name, prompt_type, self.prompts_dict
        )
        # import there due to circular imports
        from mteb import get_task

        task = get_task(task_name)

        # to passage prompts won't be applied to passages
        if not self.apply_instruction_to_passages and prompt_type == PromptType.passage:
            instruction = None
            logger.info(
                f"No instruction used, because prompt type = {prompt_type.passage}"
            )

        if task.metadata.type in ["STS", "PairClassification", "Summarization"]:
            logger.info(
                f"No instruction used, because task type = {task.metadata.type}"
            )
            instruction = None

        if instruction:
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


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

HIT_TMG_task_prompts = {
    "AmazonCounterfactualClassification": "Given an Amazon review, judge whether it is counterfactual.",
    "AmazonPolarityClassification": "Classifying Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classifying the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given an online banking query, find the corresponding intents",
    "EmotionClassification": "Classifying the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classifying the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classifying the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classifying the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classifying the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classifying the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Categorizing the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classifying sentiment of the customer review for iPhone into positive or negative",
    "OnlineShopping": "Classifying sentiment of the customer review into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    "MasakhaNEWSClassification": "Classifying the category of french news.",
    "CBD": "Classifying the sentiment of polish tweet reviews",
    "PolEmo2.0-IN": "Classifying the sentiment of in-domain (medicine and hotels) online reviews",
    "PolEmo2.0-OUT": "Classifying the sentiment of out-of-domain (products and school) online reviews",
    "AllegroReviews": "Classifying the sentiment of reviews from e-commerce marketplace Allegro",
    "PAC": 'Classifying the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "GeoreviewClassification": "Classifying the sentiment of Russian reviews.",
    "HeadlineClassification": "Classifying the topic of Russian headlines.",
    "InappropriatenessClassification": "Detecting inappropriate messages on sensitive topics",
    "KinopoiskClassification": "Classifying the sentiment of Kinopoisk reviews.",
    "RuReviewsClassification": "Classifying the sentiment of Russian product reviews.",
    "RuSciBenchGRNTIClassification": "Classifying the topic of Russian scientific papers.",
    "RuSciBenchOECDClassification": "Classifying the topic of Russian scientific papers.",
    "CEDRClassification": "Classification of sentences by emotions.",
    "SensitiveTopicsClassification": "Detecting inappropriate messages on sensitive topics.",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
    "AlloProfClusteringS2S": "Identify the main category of Allo Prof document based on the titles",
    "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
    "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles",
    "EightTagsClustering": "Identify of headlines from social media posts in Polish into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
    "GeoreviewClusteringP2P": "Identify the topic or theme of the Russian reviews.",
    "RuSciBenchGRNTIClusteringP2P": "Identify the topic or theme of the Russian articles.",
    "RuSciBenchOECDClusteringP2P": "Identify the topic or theme of the Russian articles.",
}

HIT_TMG_INSTRUCTION = "Instruct: {instruction} \n Query: "

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1 = ModelMeta(
    loader=partial(  # type: ignore
        KALMWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
        revision="45e42c89990c40aca042659133fc8b13c28634b5",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=False,
        prompts_dict=HIT_TMG_task_prompts,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    revision="45e42c89990c40aca042659133fc8b13c28634b5",
    release_date="2024-10-23",
    languages=["eng-Latn", "zho-Hans"],
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
    languages=["eng-Latn", "zho-Hans"],
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
        KALMWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=False,
        prompts_dict=HIT_TMG_task_prompts,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
    release_date="2024-12-26",
    languages=["eng-Latn", "zho-Hans"],
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
