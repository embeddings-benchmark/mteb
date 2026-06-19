"""OpenAI-compatible API Wrappers for MTEB.

This module provides wrappers for embedding and reranking models served via
OpenAI-compatible APIs (such as vLLM servers, OpenAI, or other compatible servers).

Classes:
    OpenAIBaseWrapper: Base class with shared HTTP connection logic
    OpenAIAPIWrapper: Wrapper for embedding models using /v1/embeddings endpoint
    OpenAIRerankWrapper: Wrapper for reranking models using /v1/rerank endpoint

Examples:
    Embeddings with vLLM:
        >>> from mteb.models import OpenAIAPIWrapper
        >>> wrapper = OpenAIAPIWrapper(
        ...     endpoint_url="http://localhost:8000",
        ...     model_name="BAAI/bge-small-en-v1.5"
        ... )

    Reranking with vLLM:
        >>> from mteb.models import OpenAIRerankWrapper
        >>> wrapper = OpenAIRerankWrapper(
        ...     endpoint_url="http://localhost:8001",
        ...     model_name="BAAI/bge-reranker-v2-m3"
        ... )

    With OpenAI API:
        >>> wrapper = OpenAIAPIWrapper(
        ...     endpoint_url="https://api.openai.com/v1",
        ...     model_name="text-embedding-3-small",
        ...     api_key="sk-..."
        ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

logger = logging.getLogger(__name__)


class OpenAIBaseWrapper:
    """Base class for OpenAI-compatible API wrappers.

    Provides shared HTTP connection, retry logic, and error handling.

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the model to use
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """Initialize the base wrapper."""
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

    def _make_request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP POST request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/v1/embeddings")
            payload: JSON payload to send

        Returns:
            JSON response from the server

        Raises:
            RuntimeError: If all retries fail
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint_url}{endpoint}",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request timeout (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying..."
                    )
                    continue
                raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e}. Retrying..."
                    )
                    continue
                raise RuntimeError(f"Failed to get response from server: {e}") from e

        # This should never be reached due to the raise above, but mypy needs it
        raise RuntimeError("Failed to get response after all retries")

    def _verify_server(self) -> None:
        """Verify that the server is reachable and get model info."""
        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/models",
                timeout=10,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            models = response.json()

            # Check if our model is available
            available_models = [m["id"] for m in models.get("data", [])]
            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found in server. "
                    f"Available models: {available_models}"
                )
                # Still allow initialization - model name might be alias
                return

            logger.info(f"Successfully connected to server. Model: {self.model_name}")

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to server at {self.endpoint_url}: {e}"
            ) from e


class OpenAIAPIWrapper(OpenAIBaseWrapper, AbsEncoder):
    """OpenAI-compatible API wrapper for MTEB embedding benchmarks.

    This wrapper communicates with embedding models served via OpenAI-compatible
    HTTP APIs using the /v1/embeddings endpoint.

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the model to use
        api_key: Optional API key for authentication
        prompt_dict: A dictionary mapping task names to prompt strings
        use_instructions: Whether to use instructions from the prompt_dict
        instruction_template: A template or callable to format instructions
        apply_instruction_to_documents: Whether to apply instructions to
            documents (passages). Default True.
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        batch_size: Default batch size for processing embeddings (default: 32).
            Can be overridden per encode() call.
        verify_ssl: Whether to verify SSL certificates (default: True)
        max_length: Maximum sequence length for truncation. If None,
            auto-detected from model metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        prompt_dict: dict[str, str] | None = None,
        use_instructions: bool = False,
        instruction_template: (
            str | Callable[[str, PromptType | None], str] | None
        ) = None,
        apply_instruction_to_documents: bool = True,
        timeout: int = 300,
        max_retries: int = 3,
        batch_size: int = 32,
        verify_ssl: bool = True,
        max_length: int | None = None,
    ):
        """Initialize the OpenAI API wrapper for embeddings."""
        # Initialize base class
        super().__init__(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )

        # Embedding-specific attributes
        self.prompts_dict = prompt_dict
        self.use_instructions = use_instructions
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_documents
        self.batch_size = batch_size
        self.max_length = max_length

        # Create model metadata for MTEB compatibility
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={"name": model_name})

        if use_instructions and instruction_template is None:
            raise ValueError(
                "To use instructions, an instruction_template must be provided. "
                "For example, `Instruction: {instruction}`"
            )

        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )

        # Verify server and detect max_length
        self._verify_server()
        self._detect_max_length()

    def _detect_max_length(self) -> None:
        """Auto-detect max_length from model metadata if not provided."""
        if self.max_length is not None:
            return

        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/models",
                timeout=10,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            models = response.json()

            for model in models.get("data", []):
                if model["id"] != self.model_name:
                    continue
                # vLLM returns max_model_len in model metadata
                max_model_len = model.get("max_model_len")
                if max_model_len:
                    self.max_length = max_model_len
                    logger.info(
                        f"Auto-detected max_length={self.max_length} from model "
                        f"metadata"
                    )
                break
        except Exception as e:
            # If we can't detect max_length, that's fine - will use model default
            logger.debug(f"Could not auto-detect max_length: {e}")

    def _get_embeddings(self, texts: list[str]) -> Array:
        """Get embeddings from the server via OpenAI-compatible API.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        # Add truncation parameter if max_length is set
        # Note: vLLM supports truncate_prompt_tokens, OpenAI uses different params
        if self.max_length:
            payload["truncate_prompt_tokens"] = self.max_length

        result = self._make_request("/v1/embeddings", payload)

        # Extract embeddings in correct order
        embeddings = [None] * len(texts)
        for item in result["data"]:
            embeddings[item["index"]] = item["embedding"]

        # Validate all embeddings were returned
        missing_indices = [i for i, emb in enumerate(embeddings) if emb is None]
        if missing_indices:
            raise RuntimeError(
                f"Incomplete embeddings from server: expected {len(texts)} "
                f"embeddings, got {len(texts) - len(missing_indices)}. "
                f"Missing indices: {missing_indices[:10]}"
            )

        # Convert to numpy array
        return np.array(embeddings, dtype=np.float32)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode the given sentences using the OpenAI-compatible API.

        Args:
            inputs: The sentences to encode
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The type of prompt (query or passage)
            **kwargs: Additional arguments (batch_size, show_progress_bar, precision)

        Returns:
            The encoded sentences as embeddings
        """
        # Get encode kwargs with defaults
        batch_size = kwargs.get("batch_size", self.batch_size)
        show_progress_bar = kwargs.get("show_progress_bar", True)

        # Determine prompt to use
        prompt = ""
        if self.use_instructions and self.prompts_dict is not None:
            prompt = self.get_task_instruction(task_metadata, prompt_type)
        elif self.prompts_dict is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            if prompt_name is not None:
                prompt = self.prompts_dict.get(prompt_name, "")

        # Skip instruction for documents if configured
        if (
            self.use_instructions
            and self.apply_instruction_to_passages is False
            and prompt_type == PromptType.document
        ):
            logger.info(f"No instruction used, because prompt type = {prompt_type}")
            prompt = ""
        elif prompt:
            logger.info(
                f"Using instruction: '{prompt}' for task: '{task_metadata.name}' "
                f"prompt type: '{prompt_type}'"
            )

        # Collect all texts from batches
        texts = [prompt + text for batch in inputs for text in batch["text"]]

        # Process in batches to avoid overwhelming the server
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding batches",
            disable=not show_progress_bar,
        ):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        return embeddings


class OpenAIRerankWrapper(OpenAIBaseWrapper):
    """OpenAI-compatible API wrapper for MTEB reranking benchmarks.

    This wrapper communicates with reranking models served via OpenAI-compatible
    HTTP APIs using the /v1/rerank endpoint.

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the reranking model to use
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        batch_size: Default batch size for processing (default: 32).
            Can be overridden per predict() call.
        verify_ssl: Whether to verify SSL certificates (default: True)
        top_k: Optional number of top results to return per query
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        timeout: int = 300,
        max_retries: int = 3,
        batch_size: int = 32,
        verify_ssl: bool = True,
        top_k: int | None = None,
    ):
        """Initialize the OpenAI Rerank wrapper."""
        # Initialize base class
        super().__init__(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )

        # Reranking-specific attributes
        self.batch_size = batch_size
        self.top_k = top_k

        # Create model metadata for MTEB compatibility
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={"name": model_name})

        # Verify server is reachable
        self._verify_server()

    def _rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> Array:
        """Get relevance scores for query-document pairs.

        Args:
            query: The query string
            documents: List of documents to rank
            top_k: Optional number of top results to return

        Returns:
            Array of relevance scores in the same order as input documents
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }

        if top_k is not None:
            payload["top_n"] = top_k

        result = self._make_request("/v1/rerank", payload)

        # Extract scores in original document order
        # The API returns documents sorted by relevance, so we need to reorder
        scores = [0.0] * len(documents)
        for item in result["results"]:
            original_index = item["index"]
            scores[original_index] = item["relevance_score"]

        return np.array(scores, dtype=np.float32)

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Predict relevance scores for query-document pairs.

        Args:
            inputs1: Queries (first input)
            inputs2: Documents (second input)
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The type of prompt
            **kwargs: Additional arguments (batch_size, show_progress_bar)

        Returns:
            Relevance scores for each query-document pair
        """
        # Get kwargs with defaults
        batch_size = kwargs.get("batch_size", self.batch_size)
        show_progress_bar = kwargs.get("show_progress_bar", True)

        # Collect all queries and documents
        queries = [text for batch in inputs1 for text in batch["text"]]
        documents = [text for batch in inputs2 for text in batch["text"]]

        # For reranking, we typically have one query with multiple documents
        # or equal numbers of queries and documents (pairwise)
        if len(queries) == 1 and len(documents) > 1:
            # One query, many documents - rank all documents for the query
            scores = self._rerank(queries[0], documents, self.top_k)
        elif len(queries) == len(documents):
            # Pairwise scoring - one query per document
            all_scores = []
            for i in tqdm(
                range(0, len(queries), batch_size),
                desc="Reranking batches",
                disable=not show_progress_bar,
            ):
                batch_queries = queries[i : i + batch_size]
                batch_docs = documents[i : i + batch_size]

                # Score each pair individually
                batch_scores = []
                for query, doc in zip(batch_queries, batch_docs):
                    score = self._rerank(query, [doc])[0]
                    batch_scores.append(score)

                all_scores.extend(batch_scores)

            scores = np.array(all_scores, dtype=np.float32)
        else:
            raise ValueError(
                f"Invalid input sizes: {len(queries)} queries and "
                f"{len(documents)} documents. Expected either 1 query with N "
                f"documents, or N queries with N documents (pairwise)."
            )

        return scores
