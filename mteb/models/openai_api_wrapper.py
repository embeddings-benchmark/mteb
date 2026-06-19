"""OpenAI API Wrapper for MTEB.

This wrapper provides HTTP-based access to embedding models served via the
OpenAI-compatible API (such as vLLM servers) for MTEB benchmarks.

Use Cases:
- Testing remote vLLM servers (CPU or GPU)
- Benchmarking production deployments with OpenAI-compatible APIs
- Reusing running instances across multiple benchmark runs
- Avoiding repeated model loading overhead

Example:
    ```python
    from mteb.models import OpenAIAPIWrapper

    # With vLLM server
    wrapper = OpenAIAPIWrapper(
        endpoint_url="http://localhost:8000",
        model_name="BAAI/bge-small-en-v1.5"
    )

    # With OpenAI API
    wrapper = OpenAIAPIWrapper(
        endpoint_url="https://api.openai.com/v1",
        model_name="text-embedding-3-small",
        api_key="sk-..."
    )
    ```

Comparison with VllmEncoderWrapper:
- VllmEncoderWrapper: Local in-process instantiation via `vllm.LLM()`
- OpenAIAPIWrapper: Remote HTTP API access via `/v1/embeddings`
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


class OpenAIAPIWrapper(AbsEncoder):
    """OpenAI-compatible API wrapper for MTEB embedding benchmarks.

    This wrapper communicates with embedding models served via OpenAI-compatible
    HTTP APIs (such as vLLM, OpenAI, or other compatible servers).

    Args:
        endpoint_url: URL of the OpenAI-compatible server
            (e.g., "http://localhost:8000" for vLLM, "https://api.openai.com/v1" for OpenAI)
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
        """Initialize the OpenAI API wrapper."""
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.prompts_dict = prompt_dict
        self.use_instructions = use_instructions
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_documents
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.verify_ssl = verify_ssl
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

        # Verify server is reachable
        self._verify_server()

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
            self._detect_max_length_from_models(models)

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to server at {self.endpoint_url}: {e}"
            ) from e

    def _detect_max_length_from_models(self, models: dict[str, Any]) -> None:
        """Auto-detect max_length from model metadata if not provided."""
        if self.max_length is not None:
            return

        for model in models.get("data", []):
            if model["id"] != self.model_name:
                continue
            # vLLM returns max_model_len in model metadata
            max_model_len = model.get("max_model_len")
            if max_model_len:
                self.max_length = max_model_len
                logger.info(
                    f"Auto-detected max_length={self.max_length} from model metadata"
                )
            break

    def _get_embeddings(self, texts: list[str]) -> Array:
        """Get embeddings from the server via OpenAI-compatible API.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        # Add truncation parameter if max_length is set
        # Note: vLLM supports truncate_prompt_tokens, OpenAI uses different params
        if self.max_length:
            payload["truncate_prompt_tokens"] = self.max_length

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint_url}/v1/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings in correct order
                embeddings = [None] * len(texts)
                for item in result["data"]:
                    embeddings[item["index"]] = item["embedding"]

                # Validate all embeddings were returned
                missing_indices = [i for i, emb in enumerate(embeddings) if emb is None]
                if missing_indices:
                    raise RuntimeError(
                        f"Incomplete embeddings from vLLM server: "
                        f"expected {len(texts)} embeddings, "
                        f"got {len(texts) - len(missing_indices)}. "
                        f"Missing indices: {missing_indices[:10]}"
                    )

                # Convert to numpy array
                return np.array(embeddings, dtype=np.float32)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request timeout "
                        f"(attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying..."
                    )
                    continue
                raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request failed "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying..."
                    )
                    continue
                raise RuntimeError(f"Failed to get embeddings from server: {e}") from e

        # This should never be reached due to the raise above, but mypy needs it
        raise RuntimeError("Failed to get embeddings after all retries")

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
                f"Using instruction: '{prompt}' for "
                f"task: '{task_metadata.name}' "
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
