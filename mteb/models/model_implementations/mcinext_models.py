import logging
import os
import time
import warnings
from typing import Any

import numpy as np
import requests

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

logger = logging.getLogger(__name__)

HAKIM_CITATION = """@article{sarmadi2025hakim,
      title={Hakim: Farsi Text Embedding Model},
      author={Sarmadi, Mehran and Alikhani, Morteza and Zinvandi, Erfan and Pourbahman, Zahra},
      journal={arXiv preprint arXiv:2505.08435},
      year={2025}
}"""

MODEL_API_NAMES = {
    "hakim": "Hakim",
    "hakim-small": "Hakim_small",
    "hakim-unsup": "Hakim_unsuper",
}

# Dataset task mappings with descriptions and task IDs
DATASET_TASKS = {
    "PersianTextEmotion": ("دسته بندی , دسته بندی احساس متن", 1),
    "PersianTextEmotion.v2": ("دسته بندی , دسته بندی احساس متن", 1),
    "PersianFoodSentimentClassification": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "SentimentDKSF": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "MassiveIntentClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "MassiveScenarioClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "StyleClassification": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerChatbotConvSAAnger": (
        "دسته بندی , تحلیل احساس عصبانیت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASatisfaction": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAFriendship": (
        "دسته بندی , تحلیل احساس صمیمیت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAFear": (
        "دسته بندی , تحلیل احساس ترس کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAJealousy": (
        "دسته بندی , تحلیل احساس حسادت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASurprise": (
        "دسته بندی , تحلیل احساس شگفتی کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSALove": (
        "دسته بندی , تحلیل احساس عشق کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASadness": (
        "دسته بندی , تحلیل احساس غصه کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAHappiness": (
        "دسته بندی , تحلیل احساس خوشحالی کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotConvSAToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "PersianTextTone": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerTextToneClassification.v3": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerChatbotToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotRAGToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotRAGToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotSatisfactionLevelClassification": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        1,
    ),
    "DigikalamagClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClassification.v2": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClassification.v2": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "DeepSentiPers": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "DeepSentiPers.v2": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "Farsick": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "Query2Query": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "SynPerSTS": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "BeytooteClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "DigikalamagClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "HamshahriClustring": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClustring": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "MIRACLReranking": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", 3),
    "WikipediaRerankingMultilingual": (
        "تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟",
        3,
    ),
    "SAMSumFa": (
        "تشخیص ارتباط , متن اول یک مکالمه است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotSumSRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotRAGSumSRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerQARetrieval": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", 3),
    "SynPerChatbotTopicsRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotRAGTopicsRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        3,
    ),
    "WebFAQRetrieval": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", 3),
}

# Add all retrieval datasets with the same instruction and task ID
RETRIEVAL_DATASETS = [
    "ArguAna-Fa",
    "ClimateFEVER-Fa",
    "CQADupstackAndroidRetrieval-Fa",
    "CQADupstackEnglishRetrieval-Fa",
    "CQADupstackGamingRetrieval-Fa",
    "CQADupstackGisRetrieval-Fa",
    "CQADupstackMathematicaRetrieval-Fa",
    "CQADupstackPhysicsRetrieval-Fa",
    "CQADupstackProgrammersRetrieval-Fa",
    "CQADupstackStatsRetrieval-Fa",
    "CQADupstackTexRetrieval-Fa",
    "CQADupstackUnixRetrieval-Fa",
    "CQADupstackWebmastersRetrieval-Fa",
    "CQADupstackWordpressRetrieval-Fa",
    "DBPedia-Fa",
    "FiQA2018-Fa",
    "HotpotQA-Fa",
    "MSMARCO-Fa",
    "NFCorpus-Fa",
    "NQ-Fa",
    "QuoraRetrieval-Fa",
    "SCIDOCS-Fa",
    "SciFact-Fa",
    "TRECCOVID-Fa",
    "Touche2020-Fa",
    "MIRACLRetrieval",
    "WikipediaRetrievalMultilingual",
    "MIRACLRetrievalHardNegatives",
    "HotpotQA-FaHardNegatives",
    "MSMARCO-FaHardNegatives",
    "NQ-FaHardNegatives",
    "FEVER-FaHardNegatives",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023RetrievalHardNegatives",
    "ArguAna-Fa.v2",
    "FiQA2018-Fa.v2",
    "SCIDOCS-Fa.v2",
    "SciFact-Fa.v2",
    "TRECCOVID-Fa.v2",
    "Touche2020-Fa.v2",
    "PersianWebDocumentRetrieval",
    "SynPerChatbotRAGFAQRetrieval",
]

for dataset in RETRIEVAL_DATASETS:
    DATASET_TASKS[dataset] = ("تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟", 3)


class APIError(Exception):
    """Custom exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(
            f"API Error: {message} (Status Code: {status_code})"
            if status_code
            else f"API Error: {message}"
        )
        self.status_code = status_code


class HakimModelWrapper(AbsEncoder):
    """A simplified wrapper for the Hakim instruction-following model."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        api_model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        **kwargs: Any,
    ):
        self.model_name = api_model_name
        self.api_url = f"https://mcinext.ai/api/{api_model_name}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = os.getenv("MCINEXT_API_KEY")
        if not self.api_key:
            raise ValueError("MCINEXT_API_KEY environment variable not set.")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        logger.info(f"Initialized model wrapper for: {api_model_name}")

    def _preprocess_sample(
        self,
        sample: str,
        task_name: str,
        prompt_type: PromptType | None,
        sub: str | None,
    ) -> str:
        """Preprocesses a single text sample based on the task."""
        if "unsup" in self.model_name:
            return sample

        task_prompt, task_id = DATASET_TASKS.get(task_name, (None, None))

        if not task_prompt:
            msg = f"Unknown dataset: {task_name}, no preprocessing applied."
            logger.warning(msg)
            warnings.warn(msg)
            return sample

        task_prompt = f"مسئله : {task_prompt}"

        if task_id == 1:
            return f"{task_prompt} | متن : {sample}"
        if task_id == 3:
            if sub == "sentence1" or (prompt_type and prompt_type.value == "query"):
                return f"{task_prompt} | متن اول : {sample}"
            if sub == "sentence2" or (prompt_type and prompt_type.value == "document"):
                return f"{task_prompt} | متن دوم : {sample}"
        return sample

    def _make_api_request(self, data: dict[str, Any]) -> list[list[float]]:
        """Makes an API request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=data, timeout=60
                )
                response.raise_for_status()
                response_data = response.json()

                if not response_data.get("data") or not all(
                    "embedding" in item for item in response_data["data"]
                ):
                    raise APIError("Invalid response format from API.")

                return [item["embedding"] for item in response_data["data"]]

            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response else None
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    raise APIError(f"Client error: {e}", status_code)
                time.sleep(self.retry_delay * (2**attempt))

        raise APIError(f"API request failed after {self.max_retries} attempts.")

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes sentences using the API.

        Returns:
            A 2D numpy array of shape (num_sentences, embedding_dim) containing the embeddings.
        """
        if not sentences or not all(isinstance(s, str) for s in sentences):
            raise ValueError("Input must be a non-empty list of strings.")

        logger.info(
            f"Starting encoding for {len(sentences)} sentences, task: {task_name}, batch_size: {batch_size}"
        )

        sub = kwargs.get("sub")
        processed_sentences = [
            self._preprocess_sample(s, task_name, prompt_type, sub) for s in sentences
        ]

        all_embeddings = []
        for i in range(0, len(processed_sentences), batch_size):
            batch = processed_sentences[i : i + batch_size]
            data = {
                "model": MODEL_API_NAMES[self.model_name],
                "input": batch,
                "encoding_format": "float",
                "add_special_tokens": True,
            }
            try:
                batch_embeddings = self._make_api_request(data)
                if len(batch_embeddings) != len(batch):
                    raise APIError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                all_embeddings.extend(batch_embeddings)
            except APIError as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                raise e

        logger.info(
            f"Encoding completed successfully for {len(all_embeddings)} sentences."
        )
        return np.array(all_embeddings, dtype=np.float32)


hakim = ModelMeta(
    loader=HakimModelWrapper,
    loader_kwargs=dict(
        api_model_name="hakim",
        model_type=["dense"],
    ),
    name="MCINext/Hakim",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail",
        "SAMSumFa",
        "SynPerChatbotSumSRetrieval",
        "SynPerChatbotRAGSumSRetrieval",
        "SynPerChatbotConvSAClassification",
        "SynPerChatbotConvSAToneChatbotClassification",
        "SynPerChatbotConvSAToneUserClassification",
        "SynPerChatbotSatisfactionLevelClassification",
        "SynPerChatbotRAGToneChatbotClassification",
        "SynPerChatbotRAGToneUserClassification",
        "SynPerChatbotToneChatbotClassification",
        "SynPerChatbotToneUserClassification",
        "SynPerTextToneClassification",
        "SIDClassification",
        "PersianTextEmotion",
        "SentimentDKSF",
        "NLPTwitterAnalysisClassification",
        "DigikalamagClassification",
        "DigikalamagClustering",
        "NLPTwitterAnalysisClustering",
        "SIDClustring",
        "CExaPPC",
        "SynPerChatbotRAGFAQPC",
        "FarsiParaphraseDetection",
        "SynPerTextKeywordsPC",
        "SynPerQAPC",
        "ParsinluEntail",
        "ParsinluQueryParaphPC",
        "FiQA2018-Fa",
        "HotpotQA-Fa",
        "MSMARCO-Fa",
        "NFCorpus-Fa",
        "SciFact-Fa",
        "SynPerQARetrieval",
        "SynPerChatbotTopicsRetrieval",
        "SynPerChatbotRAGTopicsRetrieval",
        "SynPerChatbotRAGFAQRetrieval",
        "Farsick",
        "SynPerSTS",
        "Query2Query",
    },
    citation=HAKIM_CITATION,
)


hakim_small = ModelMeta(
    loader=HakimModelWrapper,
    loader_kwargs=dict(
        api_model_name="hakim-small",
        model_type=["dense"],
    ),
    name="MCINext/Hakim-small",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=38_736_384,
    memory_usage_mb=148,
    embed_dim=512,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-small",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail",
        "SAMSumFa",
        "SynPerChatbotSumSRetrieval",
        "SynPerChatbotRAGSumSRetrieval",
        "SynPerChatbotConvSAClassification",
        "SynPerChatbotConvSAToneChatbotClassification",
        "SynPerChatbotConvSAToneUserClassification",
        "SynPerChatbotSatisfactionLevelClassification",
        "SynPerChatbotRAGToneChatbotClassification",
        "SynPerChatbotRAGToneUserClassification",
        "SynPerChatbotToneChatbotClassification",
        "SynPerChatbotToneUserClassification",
        "SynPerTextToneClassification",
        "SIDClassification",
        "PersianTextEmotion",
        "SentimentDKSF",
        "NLPTwitterAnalysisClassification",
        "DigikalamagClassification",
        "DigikalamagClustering",
        "NLPTwitterAnalysisClustering",
        "SIDClustring",
        "CExaPPC",
        "SynPerChatbotRAGFAQPC",
        "FarsiParaphraseDetection",
        "SynPerTextKeywordsPC",
        "SynPerQAPC",
        "ParsinluEntail",
        "ParsinluQueryParaphPC",
        "FiQA2018-Fa",
        "HotpotQA-Fa",
        "MSMARCO-Fa",
        "NFCorpus-Fa",
        "SciFact-Fa",
        "SynPerQARetrieval",
        "SynPerChatbotTopicsRetrieval",
        "SynPerChatbotRAGTopicsRetrieval",
        "SynPerChatbotRAGFAQRetrieval",
        "Farsick",
        "SynPerSTS",
        "Query2Query",
    },
    citation=HAKIM_CITATION,
)

hakim_unsup = ModelMeta(
    loader=HakimModelWrapper,
    loader_kwargs=dict(
        api_model_name="hakim-unsup",
        model_type=["dense"],
    ),
    name="MCINext/Hakim-unsup",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail",
        "Farsick",
        "MSMARCO-Fa",
        "Query2Query",
    },
    citation=HAKIM_CITATION,
)
