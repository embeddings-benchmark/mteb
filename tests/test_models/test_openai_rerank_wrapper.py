"""Tests for OpenAIAPIRerankWrapper.

All tests are fully mocked and do not require a running server.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from mteb.models.openai_wrappers import OpenAIAPIRerankWrapper


class TestOpenAIAPIRerankWrapper:
    """Test OpenAIAPIRerankWrapper functionality."""

    @patch("requests.get")
    def test_initialization(self, mock_get):
        """Test that wrapper initializes and connects to server."""
        # Mock the /v1/models endpoint
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-reranker",
                }
            ]
        }
        mock_get.return_value = mock_response

        wrapper = OpenAIAPIRerankWrapper(
            endpoint_url="http://localhost:8001",
            model_name="test-reranker",
        )
        assert wrapper.endpoint_url == "http://localhost:8001"
        assert wrapper.mteb_model_meta.name == "test-reranker"

    def test_invalid_endpoint(self):
        """Test that invalid endpoint raises ConnectionError."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection failed"
            )
            with pytest.raises(ConnectionError):
                OpenAIAPIRerankWrapper(
                    endpoint_url="http://invalid-endpoint:9999",
                    model_name="test-model",
                )

    @patch("requests.post")
    @patch("requests.get")
    def test_basic_reranking(self, mock_get, mock_post):
        """Test basic reranking functionality with mocked responses."""
        # Mock server initialization
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"data": [{"id": "test-reranker"}]}
        mock_get.return_value = mock_get_response

        # Mock reranking response
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.75},
                {"index": 1, "relevance_score": 0.05},
            ]
        }
        mock_post.return_value = mock_post_response

        wrapper = OpenAIAPIRerankWrapper(
            endpoint_url="http://localhost:8001",
            model_name="test-reranker",
        )

        # Test reranking
        query = "What is machine learning?"
        documents = ["ML is AI", "I like pizza", "Deep learning uses neural nets"]
        scores = wrapper._rerank(query, documents)

        # Verify output
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (3,)
        assert scores.dtype == np.float32
        # Check scores are in original document order
        assert scores[0] == pytest.approx(0.95)
        assert scores[1] == pytest.approx(0.05)
        assert scores[2] == pytest.approx(0.75)

    @patch("requests.post")
    @patch("requests.get")
    def test_predict_pairwise(self, mock_get, mock_post):
        """Test predict with pairwise query-document scoring."""
        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        # Mock server
        mock_get.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})
        )

        # Mock will be called twice (once for each pair)
        mock_post.side_effect = [
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={"results": [{"index": 0, "relevance_score": 0.8}]}
                ),
            ),
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={"results": [{"index": 0, "relevance_score": 0.3}]}
                ),
            ),
        ]

        wrapper = OpenAIAPIRerankWrapper(
            endpoint_url="http://localhost:8001",
            model_name="test",
        )

        # Two queries, two documents (pairwise)
        query_dataloader = [{"text": ["query1", "query2"]}]
        doc_dataloader = [{"text": ["doc1", "doc2"]}]

        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "test"

        scores = wrapper.predict(
            query_dataloader,
            doc_dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
            show_progress_bar=False,
        )

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)
        assert scores[0] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(0.3)

    @patch("requests.post")
    @patch("requests.get")
    def test_top_k_parameter(self, mock_get, mock_post):
        """Test that top_k parameter can be passed to predict()."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        # Mock reranking response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"results": [{"index": 0, "relevance_score": 0.8}]}
            ),
        )

        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        wrapper = OpenAIAPIRerankWrapper(
            endpoint_url="http://localhost:8001",
            model_name="test-model",
        )

        # Test that top_k can be passed to predict()
        query_dataloader = [{"text": ["query1"]}]
        doc_dataloader = [{"text": ["doc1"]}]

        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "test"

        scores = wrapper.predict(
            query_dataloader,
            doc_dataloader,
            task_metadata=mock_metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
            show_progress_bar=False,
            top_k=10,
        )

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (1,)

    @patch("requests.post")
    @patch("requests.get")
    def test_predict_invalid_sizes(self, mock_get, mock_post):
        """Test predict with invalid input sizes raises ValueError."""
        from mteb.abstasks.task_metadata import TaskMetadata
        from mteb.types import PromptType

        mock_get.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})
        )

        wrapper = OpenAIAPIRerankWrapper(
            endpoint_url="http://localhost:8001",
            model_name="test",
        )

        # 2 queries, 3 documents - invalid (lengths don't match)
        query_dataloader = [{"text": ["query1", "query2"]}]
        doc_dataloader = [{"text": ["doc1", "doc2", "doc3"]}]

        mock_metadata = MagicMock(spec=TaskMetadata)
        mock_metadata.name = "test"

        with pytest.raises(ValueError, match="Expected equal number"):
            wrapper.predict(
                query_dataloader,
                doc_dataloader,
                task_metadata=mock_metadata,
                hf_split="test",
                hf_subset="default",
                prompt_type=PromptType.query,
            )
