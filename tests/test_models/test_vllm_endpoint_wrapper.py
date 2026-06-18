"""Tests for VllmEndpointWrapper.

These tests verify the wrapper's behavior but require a running vLLM server.
They are skipped by default unless VLLM_ENDPOINT_URL is set.
"""

import os

import numpy as np
import pytest

from mteb.models.vllm_endpoint_wrapper import VllmEndpointWrapper


@pytest.fixture
def vllm_endpoint():
    """Get vLLM endpoint URL from environment."""
    endpoint = os.getenv("VLLM_ENDPOINT_URL")
    if not endpoint:
        pytest.skip("VLLM_ENDPOINT_URL not set - skipping endpoint tests")
    return endpoint


@pytest.fixture
def vllm_model():
    """Get vLLM model name from environment."""
    return os.getenv("VLLM_MODEL_NAME", "BAAI/bge-small-en-v1.5")


class TestVllmEndpointWrapper:
    """Test VllmEndpointWrapper functionality."""

    def test_initialization(self, vllm_endpoint, vllm_model):
        """Test that wrapper initializes and connects to server."""
        wrapper = VllmEndpointWrapper(
            endpoint_url=vllm_endpoint,
            model_name=vllm_model,
        )
        assert wrapper.endpoint_url == vllm_endpoint.rstrip("/")
        assert wrapper.model_name == vllm_model

    def test_invalid_endpoint(self):
        """Test that invalid endpoint raises ConnectionError."""
        with pytest.raises(ConnectionError):
            VllmEndpointWrapper(
                endpoint_url="http://invalid-endpoint:9999",
                model_name="test-model",
            )

    def test_instruction_template_validation(self, vllm_endpoint, vllm_model):
        """Test instruction template validation."""
        # Missing instruction template when use_instructions=True
        with pytest.raises(ValueError, match="instruction_template must be"):
            VllmEndpointWrapper(
                endpoint_url=vllm_endpoint,
                model_name=vllm_model,
                use_instructions=True,
            )

        # Template missing {instruction} placeholder
        with pytest.raises(ValueError, match="must contain.*instruction"):
            VllmEndpointWrapper(
                endpoint_url=vllm_endpoint,
                model_name=vllm_model,
                use_instructions=True,
                instruction_template="query: ",
            )

    def test_basic_encoding(self, vllm_endpoint, vllm_model):
        """Test basic text encoding functionality."""
        from unittest.mock import MagicMock

        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        wrapper = VllmEndpointWrapper(
            endpoint_url=vllm_endpoint,
            model_name=vllm_model,
        )

        # Create mock dataloader
        texts = ["Hello world", "Test sentence"]
        mock_dataloader = [{"text": texts}]

        # Mock task metadata
        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "test_task"

        # Encode
        embeddings = wrapper.encode(
            mock_dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
        )

        # Verify output
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.dtype == np.float32

    def test_ssl_verification(self, vllm_model):
        """Test SSL verification can be disabled."""
        # Should not raise even with invalid SSL if verify_ssl=False
        wrapper = VllmEndpointWrapper(
            endpoint_url="https://self-signed-cert:8000",
            model_name=vllm_model,
            verify_ssl=False,
        )
        assert wrapper.verify_ssl is False

    def test_api_key_header(self, vllm_endpoint, vllm_model):
        """Test that API key is included in headers."""
        wrapper = VllmEndpointWrapper(
            endpoint_url=vllm_endpoint,
            model_name=vllm_model,
            api_key="test-key-123",
        )
        assert wrapper.api_key == "test-key-123"
