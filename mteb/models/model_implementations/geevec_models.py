from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

GEEVEC_INSTRUCTION = "Instruct: {instruction}\nQuery: "
GEEVEC_MAX_SEQ_LENGTH = 512
GEEVEC_API_DEFAULT_BATCH_SIZE = 32
GEEVEC_API_MODEL_BY_DOMAIN = {
    "coding": "geevec-embeddings-coding-1.0",
    "reasoning": "geevec-embeddings-reasoning-1.0",
    "general": "geevec-embeddings-general-1.0",
}
GEEVEC_API_INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"
GEEVEC_BRIGHT_REASONING_SUBSETS = {
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
}
GEEVEC_BRIGHT_V1_1_REASONING_TASKS = {
    "BrightAopsRetrieval",
    "BrightBiologyRetrieval",
    "BrightBiologyLongRetrieval",
    "BrightEarthScienceRetrieval",
    "BrightEarthScienceLongRetrieval",
    "BrightEconomicsRetrieval",
    "BrightEconomicsLongRetrieval",
    "BrightLeetcodeRetrieval",
    "BrightPonyRetrieval",
    "BrightPonyLongRetrieval",
    "BrightPsychologyRetrieval",
    "BrightPsychologyLongRetrieval",
    "BrightRoboticsRetrieval",
    "BrightRoboticsLongRetrieval",
    "BrightStackoverflowRetrieval",
    "BrightStackoverflowLongRetrieval",
    "BrightSustainableLivingRetrieval",
    "BrightSustainableLivingLongRetrieval",
    "BrightTheoremQAQuestionsRetrieval",
    "BrightTheoremQATheoremsRetrieval",
}


def _resolve_geevec_embeddings_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    lower = normalized.lower()

    if lower.endswith("/openapi/v1"):
        return f"{normalized}/embeddings"
    if lower.endswith("/openapi"):
        return f"{normalized}/v1/embeddings"
    return f"{normalized}/openapi/v1/embeddings"


def _resolve_geevec_domain(
    task_metadata: TaskMetadata,
    hf_subset: str,
    explicit_domain: str | None = None,
) -> str | None:
    if explicit_domain:
        return explicit_domain

    eval_langs = task_metadata.eval_langs
    if isinstance(eval_langs, dict):
        languages = [lang for langs in eval_langs.values() for lang in langs]
    else:
        languages = eval_langs

    if task_metadata.name in GEEVEC_BRIGHT_V1_1_REASONING_TASKS or (
        task_metadata.name in {"BrightRetrieval", "BrightLongRetrieval"}
        and hf_subset in GEEVEC_BRIGHT_REASONING_SUBSETS
    ):
        return "reasoning"

    task_subtypes = task_metadata.task_subtypes or []
    if (
        "Programming" in (task_metadata.domains or [])
        or "Code retrieval" in task_subtypes
        or any(lang.endswith("-Code") for lang in languages)
    ):
        return "coding"
    return None


# copied from https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
PROMPTS_DICT = {
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    "ArguAna": {"query": "Given a claim, find documents that refute the claim", "passage": "Given a claim, find documents that refute the claim"},
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
    "SICKFr": "Retrieve semantically similar text",
    "SummEvalFr": "Given a news summary, retrieve other semantically similar summaries",
    "MasakhaNEWSClassification":  "Classify the News in the given texts into one of the seven category: politics,sports,health,business,entertainment,technology,religion ",
    "OpusparcusPC":"Retrieve semantically similar text",
    "PawsX":"Retrieve semantically similar text",
    "SyntecReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofRetrieval": "Given a question, retrieve passages that answer the question",
    "BSARDRetrieval": "Given a question, retrieve passages that answer the question",
    "SyntecRetrieval": "Given a question, retrieve passages that answer the question",
    "XPQARetrieval": "Given a question, retrieve passages that answer the question",
    "MintakaRetrieval": "Given a question, retrieve passages that answer the question",
    "SICK-E-PL": "Retrieve semantically similar text",
    "SICK-R-PL": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "AFQMC": "Retrieve semantically similar text",
    "AFQMC": "Retrieve semantically similar text",
    "BQ": "Retrieve semantically similar text",
    "LCQMC": "Retrieve semantically similar text",
    "PAWSX": "Retrieve semantically similar text",
    "QBQTC": "Retrieve semantically similar text",
    "STS12": "Retrieve semantically similar text",
    "PPC": "Retrieve semantically similar text",
    "CDSC-E": "Retrieve semantically similar text",
    "PSC": "Retrieve semantically similar text",
    "ArguAna-PL": "Given a claim, find documents that refute the claim",
    "DBPedia-PL": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FiQA-PL": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question",
    "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question",
    "Quora-PL": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS-PL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim",
    "TRECCOVID-PL": "Given a query on COVID-19, retrieve documents that answer the query",
    "TERRa": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "RuBQReranking": "Given a question, retrieve Wikipedia passages that answer the question",
    "RiaNewsRetrieval": "Given a headline, retrieval relevant articles",
    "RuBQRetrieval": "Given a question, retrieve Wikipedia passages that answer the question",
    "RUParaPhraserSTS": "Retrieve semantically similar text",
    "RuSTSBenchmarkSTS": "Retrieve semantically similar text",
    "AppsRetrieval": "Given a code contest problem description, retrieve relevant code that can help solve the problem.",
    "COIRCodeSearchNetRetrieval": "Given a piece of code, retrieve the document string that summarizes the code.",
    "CodeEditSearchRetrieval": "Given a piece of code, retrieval code that in the ",
    "CodeFeedbackMT": "Given a multi-turn conversation history that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "CodeFeedbackST": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "CodeSearchNetCCRetrieval": "Given a piece of code segment, retrieve the code segment that is the latter part of the code.",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeTransOceanContest": "Given a piece of Python code, retrieve C++ code that is semantically equivalent to the input code.",
    "CodeTransOceanDL": "Given a piece of code, retrieve code that is semantically equivalent to the input code.",
    "CosQA": "Given a web search query, retrieve relevant code that can help answer the query.",
    "StackOverflowQA": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "SyntheticText2SQL": "Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
    "BibleNLPBitextMining": "Retrieve parallel sentences",
    "BUCC.v2": "Retrieve parallel sentences",
    "DiaBlaBitextMining": "Retrieve parallel sentences",
    "FloresBitextMining": "Retrieve parallel sentences",
    "IN22GenBitextMining": "Retrieve parallel sentences",
    "IndicGenBenchFloresBitextMining": "Retrieve parallel sentences",
    "NollySentiBitextMining": "Retrieve parallel sentences",
    "NTREXBitextMining": "Retrieve parallel sentences",
    "NusaTranslationBitextMining": "Retrieve parallel sentences",
    "NusaXBitextMining": "Retrieve parallel sentences",
    "Tatoeba": "Retrieve parallel sentences",
    "CTKFactsNLI": "Retrieve semantically similar text",
    "indonli": "Retrieve semantically similar text",
    "ArmenianParaphrasePC": "Retrieve semantically similar text",
    "RTE3": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
    "PpcPC": "Retrieve semantically similar text",
    "GermanSTSBenchmark": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "FaroeseSTS": "Retrieve semantically similar text",
    "FinParaSTS": "Retrieve semantically similar text",
    "JSICK": "Retrieve semantically similar text",
    "IndicCrosslingualSTS": "Retrieve semantically similar text",
    "SemRel24STS": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22.v2": "Retrieve semantically similar text",
    "STSES": "Retrieve semantically similar text",
    "STSB": "Retrieve semantically similar text",
    "AILAStatutes": "Given a situation, retrieve legal documents that are relevant to the situation.",
    # "AILAStatutes": "Identifying the most relevant statutes for a given situation",
    "HagridRetrieval": "Retrieval the relevant passage for the given query",
    "LegalBenchCorporateLobbying": "Retrieval the relevant passage for the given query",
    "LEMBPasskeyRetrieval": "Retrieval the relevant passage for the given query",
    "BelebeleRetrieval": "Retrieval the relevant passage for the given query",
    "MLQARetrieval": "Retrieval the relevant passage for the given query",
    "StatcanDialogueDatasetRetrieval": "Retrieval the relevant passage for the given query",
    "WikipediaRetrievalMultilingual": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "WebLINXCandidatesReranking": "Retrieval the relevant passage for the given query",
    "WikipediaRerankingMultilingual": "Retrieval the relevant passage for the given query",
    "STS15": "Retrieve semantically similar text",
    "MIRACLRetrievalHardNegatives": "Retrieval relevant passage for the given query",
    "BIOSSES": "Retrieve semantically similar text",
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval": {"query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question", "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question"},
    "CQADupstackUnixRetrieval": {"query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question", "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question"},
    "STS16": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    "ATEC": "Retrieve semantically similar text"
}



class GeeVecLiteModel(InstructSentenceTransformerModel):
    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        domain = _resolve_geevec_domain(task_metadata, hf_subset, kwargs.get("domain"))
        if domain is not None:
            kwargs["domain"] = domain
        else:
            kwargs.pop("domain", None)

        if (
            not self.apply_instruction_to_passages
            and prompt_type == PromptType.document
        ):
            instruction = None
        else:
            instruction = self.get_task_instruction(task_metadata, prompt_type)

        if instruction:
            logger.info(
                "Using instruction for task=%s prompt_type=%s domain=%s",
                task_metadata.name,
                prompt_type,
                domain or "general(default)",
            )

        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


class GeeVecAPIModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        timeout: int = 60,
        model_prompts: dict[str, str] | None = None,
        api_model_name: str = "geevec-embeddings-1.0",
        prompt_template: str = GEEVEC_API_INSTRUCTION_TEMPLATE,
        assert_prompts_exist: bool = True,
        apply_instruction_to_passages: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
        session: Any | None = None,
        **kwargs,
    ) -> None:
        import requests

        self._max_retries = max_retries
        self._timeout = timeout
        self._default_api_model_name = api_model_name
        self._api_model_name_override = os.environ.get("GEEVEC_API_MODEL")
        self._route_by_domain = os.environ.get("GEEVEC_ROUTE_BY_DOMAIN", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._encoding_format = os.environ.get("GEEVEC_ENCODING_FORMAT", "float")
        # Guard against context overflow on very long retrieval passages.
        self._max_input_chars = int(os.environ.get("GEEVEC_API_MAX_INPUT_CHARS", "32000"))
        self._min_input_chars = int(os.environ.get("GEEVEC_API_MIN_INPUT_CHARS", "2048"))
        self._prompt_template = prompt_template
        self._assert_prompts_exist = assert_prompts_exist
        self._apply_instruction_to_passages = apply_instruction_to_passages
        self.prompts_dict = PROMPTS_DICT
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        self._base_url = (
            base_url
            or os.environ.get("GEEVEC_BASE_URL")
            or os.environ.get("GEEKNOW_BASE_URL")
        )
        if not self._base_url:
            raise ValueError(
                "GeeVec API base URL is required. Set `GEEVEC_BASE_URL` or `GEEKNOW_BASE_URL`."
            )

        self._api_key = (
            api_key
            or os.environ.get("GEEVEC_API_KEY")
            or os.environ.get("GEEKNOW_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "GeeVec API key is required. Set `GEEVEC_API_KEY` or `GEEKNOW_API_KEY`."
            )

        self._endpoint = _resolve_geevec_embeddings_endpoint(self._base_url)
        self._session = requests.Session() if session is None else session
        self._request_exception_cls = requests.exceptions.RequestException

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        sentences = [text for batch in inputs for text in batch["text"]]

        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        input_type = self.model_prompts.get(prompt_name, "document")
        effective_domain = _resolve_geevec_domain(
            task_metadata, hf_subset, kwargs.get("domain")
        )
        sentences = self._prepare_texts(sentences, task_metadata, prompt_type)

        # Keep a safe default because the GeeVec API has a request batch limit.
        max_batch_size = kwargs.get("batch_size", GEEVEC_API_DEFAULT_BATCH_SIZE)
        api_model_name = self._resolve_api_model_name(effective_domain)

        mask_sents = [(i, t) for i, t in enumerate(sentences) if t.strip()]
        mask, no_empty_sent = list(zip(*mask_sents)) if mask_sents else ([], [])

        no_empty_embeddings = []
        for i in range(0, len(no_empty_sent), max_batch_size):
            sublist = list(no_empty_sent[i : i + max_batch_size])
            no_empty_embeddings.extend(
                self._embed_batch(
                    sublist,
                    input_type=input_type,
                    domain=effective_domain,
                    api_model_name=api_model_name,
                )
            )

        no_empty_embeddings = np.array(no_empty_embeddings, dtype=np.float32)

        if no_empty_embeddings.size > 0:
            out_dim = int(no_empty_embeddings.shape[1])
        else:
            embed_dim = self.mteb_model_meta.embed_dim
            if not isinstance(embed_dim, int):
                raise ValueError("GeeVec API model embed_dim must be an int.")
            out_dim = embed_dim

        meta_embed_dim = self.mteb_model_meta.embed_dim
        if (
            isinstance(meta_embed_dim, int)
            and no_empty_embeddings.size > 0
            and meta_embed_dim != out_dim
        ):
            logger.warning(
                "GeeVec API returned embedding dim=%s, while metadata embed_dim=%s.",
                out_dim,
                meta_embed_dim,
            )

        all_embeddings = np.zeros((len(sentences), out_dim), dtype=np.float32)
        if len(mask) > 0:
            mask = np.array(mask, dtype=int)
            all_embeddings[mask] = no_empty_embeddings
        return all_embeddings

    def _embed_batch(
        self,
        texts: list[str],
        *,
        input_type: str,
        domain: str | None,
        api_model_name: str,
    ) -> list[list[float]]:
        current_max_chars = max(self._min_input_chars, self._max_input_chars)
        req_texts = self._truncate_texts_by_chars(texts, current_max_chars)

        optional_fields = {}
        if domain is not None:
            # GeeVec API uses `domain` to route to coding/reasoning/general backends.
            optional_fields["domain"] = domain
        if input_type in {"query", "document"}:
            optional_fields["input_type"] = input_type

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries):
            payload: dict[str, Any] = {
                "input": req_texts,
                "model": api_model_name,
                "encoding_format": self._encoding_format,
            }
            req_payload = payload | optional_fields
            try:
                response = self._session.post(
                    self._endpoint,
                    headers=headers,
                    json=req_payload,
                    timeout=self._timeout,
                )
            except self._request_exception_cls as exc:
                sleep_s = min(2**attempt, 30)
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"GeeVec API network error after retries for {self._endpoint}: {exc}"
                    ) from exc
                logger.warning(
                    "GeeVec API network error (%s), retrying in %ss",
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                # Backend may return HTTP 500 with a business error for unknown model.
                if response.status_code == 500:
                    try:
                        body = response.json()
                    except Exception:
                        body = None
                    raw_message = str((body or {}).get("message", ""))
                    message = raw_message.lower()
                    if (
                        "模型不存在" in raw_message
                        or "model does not exist" in message
                        or "model not found" in message
                    ):
                        raise RuntimeError(
                            "GeeVec API model not found. "
                            f"Tried model='{api_model_name}'. "
                            "Set `GEEVEC_API_MODEL` to a valid deployed model name. "
                            f"Raw response: {response.text}"
                        )
                    if (
                        "暂无有效订阅" in raw_message
                        or "订阅套餐" in raw_message
                        or "subscription" in message
                    ):
                        raise RuntimeError(
                            "GeeVec API subscription is not active for this account/team. "
                            "Please enable a valid plan, then retry. "
                            f"Raw response: {response.text}"
                        )

                sleep_s = min(2**attempt, 30)
                logger.warning(
                    "GeeVec API temporary error (%s), retrying in %ss",
                    response.status_code,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue

            if response.status_code == 400 and optional_fields:
                # Fallback for backend versions that don't accept optional routing fields.
                optional_fields = {}
                continue

            if response.status_code == 400 and self._is_context_window_error(response.text):
                # Adaptive fallback: progressively shorten each input until it fits.
                if current_max_chars > self._min_input_chars:
                    next_max_chars = max(self._min_input_chars, int(current_max_chars * 0.8))
                    if next_max_chars < current_max_chars:
                        logger.warning(
                            "GeeVec API context window exceeded; reducing max chars per text from %s to %s and retrying.",
                            current_max_chars,
                            next_max_chars,
                        )
                        current_max_chars = next_max_chars
                        req_texts = self._truncate_texts_by_chars(texts, current_max_chars)
                        continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"GeeVec API request failed ({response.status_code}): {response.text}"
                )

            return self._parse_embeddings(response.json())

        raise RuntimeError(
            f"GeeVec API request failed after max retries for endpoint {self._endpoint}."
        )

    @staticmethod
    def _parse_embeddings(response_json: dict[str, Any]) -> list[list[float]]:
        if "data" in response_json and isinstance(response_json["data"], list):
            return [item["embedding"] for item in response_json["data"]]
        if "embeddings" in response_json and isinstance(
            response_json["embeddings"], list
        ):
            embeddings = response_json["embeddings"]
            if embeddings and isinstance(embeddings[0], dict):
                return [item["embedding"] for item in embeddings]
            return embeddings
        raise ValueError(
            "GeeVec API response does not contain embeddings in expected format."
        )

    def _resolve_api_model_name(self, domain: str | None) -> str:
        if self._api_model_name_override:
            return self._api_model_name_override

        if self._default_api_model_name != "geevec-embeddings-1.0":
            return self._default_api_model_name

        if not self._route_by_domain:
            return GEEVEC_API_MODEL_BY_DOMAIN["general"]

        if domain == "coding":
            return GEEVEC_API_MODEL_BY_DOMAIN["coding"]
        if domain == "reasoning":
            return GEEVEC_API_MODEL_BY_DOMAIN["reasoning"]
        return GEEVEC_API_MODEL_BY_DOMAIN["general"]

    def _prepare_texts(
        self,
        sentences: list[str],
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> list[str]:
        if prompt_type == PromptType.document and not self._apply_instruction_to_passages:
            return sentences

        if prompt_type == PromptType.document:
            return sentences

        instruction = self._resolve_instruction(task_metadata, prompt_type)
        if not instruction:
            if self._assert_prompts_exist:
                raise ValueError(
                    f"Prompt for task '{task_metadata.name}' not found in prompts_dict or task metadata."
                )
            return sentences

        return [self._format_query_with_instruction(instruction, text) for text in sentences]

    def _resolve_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        try:
            return self.get_instruction(task_metadata, prompt_type)
        except KeyError:
            prompt = getattr(task_metadata, "prompt", "")
            if isinstance(prompt, dict) and prompt_type:
                return str(prompt.get(prompt_type.value, ""))
            return str(prompt or "")

    def _format_query_with_instruction(self, instruction: str, text: str) -> str:
        return self._prompt_template.format(instruction=instruction, text=text)

    @staticmethod
    def _is_context_window_error(error_text: str) -> bool:
        text = (error_text or "").lower()
        return (
            "contextwindowexceeded" in text
            or "maximum context length" in text
            or "input_tokens" in text
        )

    @staticmethod
    def _truncate_texts_by_chars(texts: list[str], max_chars: int) -> list[str]:
        if max_chars <= 0:
            return [""] * len(texts)
        truncated = []
        for text in texts:
            if len(text) <= max_chars:
                truncated.append(text)
            else:
                truncated.append(text[:max_chars])
        return truncated

geevec_embeddings_1_0_lite = ModelMeta(
    loader=GeeVecLiteModel,
    loader_kwargs=dict(
        instruction_template=GEEVEC_INSTRUCTION,
        apply_instruction_to_passages=False,
        max_seq_length=GEEVEC_MAX_SEQ_LENGTH,
        prompts_dict=PROMPTS_DICT,
        model_kwargs={"dtype": torch.bfloat16},
        trust_remote_code=True,
    ),
    name="geevec-ai/geevec-embeddings-1.0-lite",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="e62d287818e2da9d647c77d6eb2e13c50e50f9eb",
    release_date="2026-04-02",
    n_parameters=366_000_000,
    n_active_parameters_override=349_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=[256, 512, 1024, 2048, 4096],
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/geevec-ai/geevec-embeddings-1.0-lite",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from=None,
    citation=None,
)

geevec_embeddings_1_0 = ModelMeta(
    loader=GeeVecAPIModel,
    loader_kwargs=dict(
        model_prompts={
            PromptType.query.value: "query",
            PromptType.document.value: "document",
        },
        api_model_name="geevec-embeddings-1.0",
    ),
    name="geevec-ai/geevec-embeddings-1.0",
    model_type=["dense"],
    languages=None,
    open_weights=False,
    revision="1",
    release_date="2026-04-02",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=4096,
    license=None,
    max_tokens=32768,
    reference="https://www.geevec.com/documentation",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from=None,
    citation=None,
)
