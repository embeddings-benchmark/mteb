"""Tests for OpenAIAPIWrapper.

All tests are fully mocked and do not require a running vLLM server.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from mteb.models.openai_api_wrapper import OpenAIAPIWrapper


class TestOpenAIAPIWrapper:
    """Test OpenAIAPIWrapper functionality."""

    @patch("requests.get")
    def test_initialization(self, mock_get):
        """Test that wrapper initializes and connects to server."""
        # Mock the /v1/models endpoint
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "max_model_len": 512,
                }
            ]
        }
        mock_get.return_value = mock_response

        wrapper = OpenAIAPIWrapper(
            endpoint_url="http://localhost:8000",
            model_name="test-model",
        )
        assert wrapper.endpoint_url == "http://localhost:8000"
        assert wrapper.mteb_model_meta.name == "test-model"
        assert wrapper.max_length == 512  # Auto-detected

    def test_invalid_endpoint(self):
        """Test that invalid endpoint raises ConnectionError."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection failed"
            )
            with pytest.raises(ConnectionError):
                OpenAIAPIWrapper(
                    endpoint_url="http://invalid-endpoint:9999",
                    model_name="test-model",
                )

    @patch("requests.get")
    def test_instruction_template_validation(self, mock_get):
        """Test instruction template validation."""
        # Mock server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        # Missing instruction template when use_instructions=True
        with pytest.raises(ValueError, match="instruction_template must be"):
            OpenAIAPIWrapper(
                endpoint_url="http://localhost:8000",
                model_name="test-model",
                use_instructions=True,
            )

        # Template missing {instruction} placeholder
        with pytest.raises(ValueError, match="must contain.*instruction"):
            OpenAIAPIWrapper(
                endpoint_url="http://localhost:8000",
                model_name="test-model",
                use_instructions=True,
                instruction_template="query: ",
            )

    @patch("requests.post")
    @patch("requests.get")
    def test_basic_encoding(self, mock_get, mock_post):
        """Test basic text encoding functionality with mocked responses."""
        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        # Mock server initialization
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_get_response

        # Mock embeddings response
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_post_response

        wrapper = OpenAIAPIWrapper(
            endpoint_url="http://localhost:8000",
            model_name="test-model",
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
            show_progress_bar=False,
        )

        # Verify output
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)
        assert embeddings.dtype == np.float32

    @patch("requests.get")
    def test_ssl_verification(self, mock_get):
        """Test SSL verification can be disabled."""
        # Mock successful response when SSL verification is disabled
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        mock_get.return_value = mock_response

        wrapper = OpenAIAPIWrapper(
            endpoint_url="https://self-signed-cert:8000",
            model_name="test-model",
            verify_ssl=False,
        )
        assert wrapper.verify_ssl is False

        # Verify that requests.get was called with verify=False
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["verify"] is False

    @patch("requests.get")
    def test_api_key_header(self, mock_get):
        """Test that API key is stored correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        wrapper = OpenAIAPIWrapper(
            endpoint_url="http://localhost:8000",
            model_name="test-model",
            api_key="test-key-123",
        )
        assert wrapper.api_key == "test-key-123"

    @patch("requests.post")
    @patch("requests.get")
    def test_encode_with_mock_task(self, mock_get, mock_post):
        """Test encoding with a mock task and actual data return."""
        # Mock server initialization
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "max_model_len": 512}]
        }
        mock_get.return_value = mock_get_response

        # Mock embeddings response - 768-dim embeddings
        # This will be called twice (once for queries, once for corpus)
        def create_embeddings(count):
            return {
                "data": [
                    {"index": j, "embedding": np.random.default_rng().random(768).tolist()}
                    for j in range(count)
                ]
            }

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200

        # Return different responses based on input size
        def mock_post_side_effect(*args, **kwargs):
            input_data = kwargs.get("json", {}).get("input", [])
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = create_embeddings(len(input_data))
            return response

        mock_post.side_effect = mock_post_side_effect

        wrapper = OpenAIAPIWrapper(
            endpoint_url="http://localhost:8000",
            model_name="test-model",
        )

        # Create a simple mock task metadata
        from mteb.abstasks.task_metadata import TaskMetadata

        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "MockRetrievalTask"
        mock_metadata.type = "Retrieval"

        # Simulate encoding queries and corpus
        queries = ["What is AI?", "How does ML work?"]
        corpus = ["AI is artificial intelligence", "ML is machine learning"]

        query_dataloader = [{"text": queries}]
        corpus_dataloader = [{"text": corpus}]

        from mteb.types import PromptType

        query_embeddings = wrapper.encode(
            query_dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
            show_progress_bar=False,
        )

        corpus_embeddings = wrapper.encode(
            corpus_dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.document,
            show_progress_bar=False,
        )

        # Verify shapes
        assert query_embeddings.shape == (2, 768)
        assert corpus_embeddings.shape == (2, 768)
        assert query_embeddings.dtype == np.float32
        assert corpus_embeddings.dtype == np.float32

    @patch("requests.post")
    @patch("requests.get")
    def test_batch_size_override(self, mock_get, mock_post):
        """Test that batch_size can be overridden in encode()."""
        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        # Mock server
        mock_get.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})
        )

        # Mock embeddings - should be called twice for batch_size=2 with 4 texts
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ]
                }
            ),
        )

        wrapper = OpenAIAPIWrapper(
            endpoint_url="http://localhost:8000",
            model_name="test",
            batch_size=32,  # Default batch size
        )

        texts = ["text1", "text2", "text3", "text4"]
        dataloader = [{"text": texts}]

        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "test"

        # Override batch_size to 2
        wrapper.encode(
            dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
            batch_size=2,
            show_progress_bar=False,
        )

        # Should be called twice (4 texts / batch_size=2)
        assert mock_post.call_count == 2
