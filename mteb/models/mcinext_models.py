from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Any

import numpy as np
import requests

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)

MODEL_API_NAMES = {
    "hakim": "Hakim",
    "hakim-small": "Hakim_small",
    "hakim-unsup": "Hakim_unsuper",
}

# Task configurations
TASK_CONFIGS = {
    "1_1": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        ["عالی", "خوب", "متوسط", "بد", "خیلی بد"],
    ),
    "1_2": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    ),
    "1_3": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    ),
    "1_4": (
        "دسته بندی , تحلیل احساس عصبانیت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_5": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_6": (
        "دسته بندی , تحلیل احساس صمیمیت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_7": ("دسته بندی , تحلیل احساس ترس کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_8": (
        "دسته بندی , تحلیل احساس حسادت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_9": (
        "دسته بندی , تحلیل احساس شگفتی کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_10": ("دسته بندی , تحلیل احساس عشق کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_11": ("دسته بندی , تحلیل احساس غصه کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_12": (
        "دسته بندی , تحلیل احساس خوشحالی کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_13": ("دسته بندی , تشخیص لحن متن", ["عامیانه", "رسمی", "کودکانه", "ادبی"]),
    "1_14": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "بازی ویدیویی",
            "راهنمای خرید",
            "سلامت و زیبایی",
            "علم و تکنولوژی",
            "عمومی",
            "هنر و سینما",
            "کتاب و ادبیات",
        ],
    ),
    "1_15": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "پزشکی",
            "کشاورزی و منابع طبیعی",
            "فنی مهندسی",
            "علوم پایه",
            "علوم انسانی",
            "هنر و معماری",
            "علمی تخصصی",
            "دامپزشکی",
        ],
    ),
    "1_16": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "هنر و طراحی",
            "مسائل اجتماعی و فعال‌سازی",
            "الهام‌بخش و انگیزشی",
            "خودرو",
            "زیبایی و لوازم آرایشی",
            "غذا و آشپزی",
            "کسب و کار و مالی",
            "مد و سبک",
            "آموزش و یادگیری",
            "علم و کشف",
            "بازی",
            "فناوری و نوآوری",
            "مذهب و معنویت",
            "حیوانات خانگی و جانوران",
            "سفر و ماجراجویی",
            "خانواده و پرورش فرزند",
            "خنده‌دار و طنز",
            "سلامت و بهزیستی",
            "خانه و باغ",
            "سیاست و مسائل روز",
            "تفریحات و فرهنگ عامه",
            "ورزش و ورزشکاری",
            "آب و هوا و فصول",
            "کتاب‌ها و ادبیات",
            "محیط زیست و پایداری",
        ],
    ),
    "1_17": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "موسیقی",
            "تقویم",
            "هشدار",
            "آب‌وهوا",
            "پخش",
            "تاریخ و زمان",
            "آشپزی",
            "ایمیل",
            "بیرون‌بر",
            "اخبار",
            "پیشنهاد",
            "فهرست‌ها",
            "اجتماعی",
            "حمل‌ونقل",
            "عمومی",
            "پرسش و پاسخ",
            "صوتی",
            "اینترنت اشیاء",
        ],
    ),
    "1_18": (
        "دسته بندی , دسته بندی احساس متن",
        ["شادی", "غم", "خشم", "انزجار", "ترس", "تعجب"],
    ),
    "1_19": ("دسته بندی , تحلیل احساس رضایت متن", ["مثبت", "منفی", "خنثی"]),
    "1_20": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        ["رسمی", "عامیانه", "کودکانه"],
    ),
    "1_21": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        ["رسمی", "عامیانه", "کودکانه"],
    ),
    "1_170": ("دسته بندی , دسته بندی موضوعی متن", []),
    "1_190": ("دسته بندی , تحلیل احساس رضایت متن", ["مثبت", "منفی"]),
    "3_1": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        None,
    ),
    "3_5": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", None),
    "3_6": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", None),
    "3_12": (
        "تشخیص ارتباط , متن اول یک مکالمه است. آیا متن دوم خلاصه ی متن اول است ؟",
        None,
    ),
    "3_13": ("تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟", None),
    "3_14": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        None,
    ),
}

# Dataset task mappings
DATASET_TASKS = {
    "PersianTextEmotion": (1, 18),
    "PersianFoodSentimentClassification": (1, 190),
    "SentimentDKSF": (1, 19),
    "MassiveIntentClassification": (1, 170),
    "MassiveScenarioClassification": (1, 17),
    "SynPerChatbotConvSAAnger": (1, 4),
    "SynPerChatbotConvSASatisfaction": (1, 5),
    "SynPerChatbotConvSAFriendship": (1, 6),
    "SynPerChatbotConvSAFear": (1, 7),
    "SynPerChatbotConvSAJealousy": (1, 8),
    "SynPerChatbotConvSASurprise": (1, 9),
    "SynPerChatbotConvSALove": (1, 10),
    "SynPerChatbotConvSASadness": (1, 11),
    "SynPerChatbotConvSAHappiness": (1, 12),
    "SynPerChatbotConvSAToneChatbotClassification": (1, 21),
    "SynPerChatbotConvSAToneUserClassification": (1, 20),
    "PersianTextTone": (1, 13),
    "SynPerChatbotToneUserClassification": (1, 2),
    "SynPerChatbotToneChatbotClassification": (1, 3),
    "SynPerChatbotRAGToneUserClassification": (1, 2),
    "SynPerChatbotRAGToneChatbotClassification": (1, 3),
    "SynPerChatbotSatisfactionLevelClassification": (1, 1),
    "DigimagClassification": (1, 14),
    "NLPTwitterAnalysisClassification": (1, 16),
    "SIDClassification": (1, 15),
    "DeepSentiPers": (1, 19),
    "DigikalamagClassification": (1, 14),
    "FarsTail": (4, 6),
    "ParsinluEntail": (4, 6),
    "ParsinluQueryParaphPC": (4, 7),
    "SynPerChatbotRAGFAQPC": (4, 1),
    "SynPerTextKeywordsPC": (4, 2),
    "SynPerQAPC": (4, 3),
    "CExaPPC": (4, 7),
    "FarsiParaphraseDetection": (4, 7),
    "Farsick": (3, 6),
    "Query2Query": (3, 6),
    "SynPerSTS": (3, 6),
    "BeytooteClustering": (1, 170),
    "DigikalamagClustering": (1, 14),
    "NLPTwitterAnalysisClustering": (1, 16),
    "HamshahriClustring": (1, 170),
    "SIDClustring": (1, 15),
    "MIRACLReranking": (3, 5),
    "WikipediaRerankingMultilingual": (3, 5),
    "SAMSumFa": (3, 12),
    "SynPerChatbotSumSRetrieval": (3, 1),
    "SynPerChatbotRAGSumSRetrieval": (3, 1),
    "SynPerQARetrieval": (3, 5),
    "SynPerChatbotTopicsRetrieval": (3, 14),
    "SynPerChatbotRAGTopicsRetrieval": (3, 14),
    "SynPerChatbotRAGFAQRetrieval": (3, 3),
    "PersianWebDocumentRetrieval": (3, 13),
}

# Add all retrieval datasets with task (3, 13)
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
]

for dataset in RETRIEVAL_DATASETS:
    DATASET_TASKS[dataset] = (3, 13)


class APIError(Exception):
    """Custom exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(
            f"API Error: {message} (Status Code: {status_code})"
            if status_code
            else f"API Error: {message}"
        )
        self.status_code = status_code


class OurInstructModelWrapper(Wrapper):
    """A simplified wrapper for the Hakim instruction-following model."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_url = f"https://mcinext.ai/api/{model_name}"
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
        logger.info(f"Initialized model wrapper for: {model_name}")

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

        task_id, subtask_id = DATASET_TASKS.get(task_name, (None, None))
        if not task_id:
            logger.warning(f"Unknown dataset: {task_name}, no preprocessing applied.")
            return sample

        task_prompt, _ = TASK_CONFIGS.get(f"{task_id}_{subtask_id}", ("", None))
        if not task_prompt:
            return sample

        task_prompt = f"مسئله : {task_prompt}"

        if task_id == 1:
            return f"{task_prompt} | متن : {sample}"
        if task_id == 3:
            if sub == "sentence1" or (prompt_type and prompt_type.value == "query"):
                return f"{task_prompt} | متن اول : {sample}"
            if sub == "sentence2" or (prompt_type and prompt_type.value == "passage"):
                return f"{task_prompt} | متن دوم : {sample}"
        return sample

    def _make_api_request(self, data: dict[str, Any]) -> list[list[float]]:
        """Makes an API request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                print(self.api_url)
                print(self.headers)
                print(data["model"])
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
        """Encodes sentences using the API."""
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
            # print(data)
            try:
                batch_embeddings = self._make_api_request(data)
                if len(batch_embeddings) != len(batch):
                    raise APIError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                all_embeddings.extend(batch_embeddings)
            except APIError as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                # Depending on desired behavior, you could return partial results or re-raise
                raise e

        logger.info(
            f"Encoding completed successfully for {len(all_embeddings)} sentences."
        )
        return np.array(all_embeddings, dtype=np.float32)


# Model metadata
hakim = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim",
        revision="v1",
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
        "FarsTail": [],
        "SAMSumFa": ["train"],
        "SynPerChatbotSumSRetrieval": ["train"],
        "SynPerChatbotRAGSumSRetrieval": ["train"],
        "SynPerChatbotConvSAClassification": ["train"],
        "SynPerChatbotConvSAToneChatbotClassification": ["train"],
        "SynPerChatbotConvSAToneUserClassification": ["train"],
        "SynPerChatbotSatisfactionLevelClassification": ["train"],
        "SynPerChatbotRAGToneChatbotClassification": ["train"],
        "SynPerChatbotRAGToneUserClassification": ["train"],
        "SynPerChatbotToneChatbotClassification": ["train"],
        "SynPerChatbotToneUserClassification": ["train"],
        "SynPerTextToneClassification": ["train"],
        "SIDClassification": ["train"],
        "PersianTextEmotion": ["train"],
        "SentimentDKSF": ["train"],
        "NLPTwitterAnalysisClassification": ["train"],
        "DigikalamagClassification": ["train"],
        "DigikalamagClustering": ["train"],
        "NLPTwitterAnalysisClustering": ["train"],
        "SIDClustring": ["train"],
        "CExaPPC": ["train"],
        "SynPerChatbotRAGFAQPC": ["train"],
        "FarsiParaphraseDetection": ["train"],
        "SynPerTextKeywordsPC": ["train"],
        "SynPerQAPC": ["train"],
        "ParsinluEntail": ["train"],
        "ParsinluQueryParaphPC": ["train"],
        "FiQA2018-Fa": ["train"],
        "HotpotQA-Fa": ["train"],
        "MSMARCO-Fa": ["train"],
        "NFCorpus-Fa": ["train"],
        "SciFact-Fa": ["train"],
        "SynPerQARetrieval": ["train"],
        "SynPerChatbotTopicsRetrieval": ["train"],
        "SynPerChatbotRAGTopicsRetrieval": ["train"],
        "SynPerChatbotRAGFAQRetrieval": ["train"],
        "Farsick": ["train"],
        "SynPerSTS": ["train"],
        "Query2Query": ["train"],
    },
)


hakim_small = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim-small",
        revision="v1",
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
        "FarsTail": [],
        "SAMSumFa": ["train"],
        "SynPerChatbotSumSRetrieval": ["train"],
        "SynPerChatbotRAGSumSRetrieval": ["train"],
        "SynPerChatbotConvSAClassification": ["train"],
        "SynPerChatbotConvSAToneChatbotClassification": ["train"],
        "SynPerChatbotConvSAToneUserClassification": ["train"],
        "SynPerChatbotSatisfactionLevelClassification": ["train"],
        "SynPerChatbotRAGToneChatbotClassification": ["train"],
        "SynPerChatbotRAGToneUserClassification": ["train"],
        "SynPerChatbotToneChatbotClassification": ["train"],
        "SynPerChatbotToneUserClassification": ["train"],
        "SynPerTextToneClassification": ["train"],
        "SIDClassification": ["train"],
        "PersianTextEmotion": ["train"],
        "SentimentDKSF": ["train"],
        "NLPTwitterAnalysisClassification": ["train"],
        "DigikalamagClassification": ["train"],
        "DigikalamagClustering": ["train"],
        "NLPTwitterAnalysisClustering": ["train"],
        "SIDClustring": ["train"],
        "CExaPPC": ["train"],
        "SynPerChatbotRAGFAQPC": ["train"],
        "FarsiParaphraseDetection": ["train"],
        "SynPerTextKeywordsPC": ["train"],
        "SynPerQAPC": ["train"],
        "ParsinluEntail": ["train"],
        "ParsinluQueryParaphPC": ["train"],
        "FiQA2018-Fa": ["train"],
        "HotpotQA-Fa": ["train"],
        "MSMARCO-Fa": ["train"],
        "NFCorpus-Fa": ["train"],
        "SciFact-Fa": ["train"],
        "SynPerQARetrieval": ["train"],
        "SynPerChatbotTopicsRetrieval": ["train"],
        "SynPerChatbotRAGTopicsRetrieval": ["train"],
        "SynPerChatbotRAGFAQRetrieval": ["train"],
        "Farsick": ["train"],
        "SynPerSTS": ["train"],
        "Query2Query": ["train"],
    },
)

hakim_unsup = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim-unsup",
        revision="v1",
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
        "FarsTail": [],
        "Farsick": ["train"],
        "MSMARCO-Fa": ["train"],
        "Query2Query": ["train"],
    },
)
