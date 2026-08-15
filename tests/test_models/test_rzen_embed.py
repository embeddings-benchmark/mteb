from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_implementations.rzen_embed_model import RzenEmbedWrapper
from mteb.types import PromptType


from mteb._create_dataloaders import _custom_collate_fn


class MockBaseModel(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setup config
        self.config = MagicMock()
        self.config.image_token_id = 151655

        # Setup base.model.embed_tokens
        self.base = MagicMock()
        self.base.config = self.config

        # embed_tokens should return a float32 tensor
        def mock_embed_tokens(input_ids):
            batch_size, seq_len = input_ids.shape
            return torch.zeros((batch_size, seq_len, 128), dtype=torch.float32)

        self.base.model.embed_tokens = mock_embed_tokens

        # visual.get_dtype()
        self.base.visual.get_dtype.return_value = torch.float32

        # visual(pixel_values, grid_thw) should return a tensor
        def mock_visual(pixel_values, grid_thw):
            # Qwen2-VL visual tower projects patch tokens
            num_tokens = pixel_values.shape[0] if len(pixel_values.shape) > 0 else 1
            return torch.zeros((num_tokens, 128), dtype=torch.float32)

        self.base.visual.side_effect = mock_visual

        # base.model(...) forward call should return an object with last_hidden_state
        def mock_forward(input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
            batch_size, seq_len, embed_dim = inputs_embeds.shape
            output = MagicMock()
            output.last_hidden_state = torch.ones((batch_size, seq_len, embed_dim), dtype=torch.float32)
            return output

        self.base.model.side_effect = mock_forward


@pytest.fixture
def mock_transformers():
    with patch("transformers.AutoConfig.from_pretrained") as mock_config, \
         patch("transformers.AutoProcessor.from_pretrained") as mock_processor, \
         patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model, \
         patch("mteb.models.modality_collators.VideoCollator") as mock_video_collator:

        # Setup mock config
        config = MagicMock()
        mock_config.return_value = config

        # Make the mock VideoCollator return the custom collate function
        mock_video_collator.side_effect = lambda *args, **kwargs: _custom_collate_fn

        # Setup mock processors
        processor = MagicMock()
        video_processor = MagicMock()

        # Mock tokenizers inside processors
        processor.tokenizer = MagicMock()
        processor.tokenizer.padding_side = "right"
        video_processor.tokenizer = MagicMock()
        video_processor.tokenizer.padding_side = "right"

        # Define processor return value depending on whether images are present
        def mock_processor_call(text, images=None, **kwargs):
            batch_size = len(text)
            seq_len = 5
            # Return dict matching Hugging Face BatchFeature structure
            features = {
                "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }
            if images is not None:
                # Insert the mock image_token_id at position 2 in input_ids
                features["input_ids"][:, 2] = 151655
                features["pixel_values"] = torch.rand((2, 10), dtype=torch.float32)
                features["image_grid_thw"] = torch.ones((batch_size, 3), dtype=torch.long)
            return features

        processor.side_effect = mock_processor_call
        video_processor.side_effect = mock_processor_call

        def select_processor(model_name, **kwargs):
            min_pixels = kwargs.get("min_pixels", 0)
            if min_pixels == 160 * 28 * 28:
                return video_processor
            return processor

        mock_processor.side_effect = select_processor

        # Setup mock model
        model_instance = MockBaseModel()
        mock_model.return_value = model_instance

        yield {
            "config": mock_config,
            "processor": mock_processor,
            "model": mock_model,
            "model_instance": model_instance,
        }


def test_rzen_embed_wrapper_text_only(mock_transformers):
    """Test encoding text-only inputs."""
    wrapper = RzenEmbedWrapper("qihoo360/RzenEmbed", device="cpu")

    # Create dummy dataloader with text column
    dataset = Dataset.from_dict({"text": ["sentence 1", "sentence 2"]})
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=_custom_collate_fn)

    # Create dummy task metadata
    task_metadata = TaskMetadata(
        name="test_task",
        description="test task",
        type="Classification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        modalities=["text"],
        prompt="dummy prompt",
        dataset={"path": "dummy", "revision": "dummy"},
    )

    embeddings = wrapper.encode(
        dataloader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 128)
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_rzen_embed_wrapper_combined_modalities(mock_transformers):
    """Test encoding combined text and image (multimodal it2i) inputs."""
    wrapper = RzenEmbedWrapper("qihoo360/RzenEmbed", device="cpu")

    # Create dummy dataset with both text and image columns
    dummy_image = Image.new("RGB", (100, 100))
    dataset = Dataset.from_dict({
        "text": ["query text 1", "query text 2"],
        "image": [dummy_image, dummy_image],
    })
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=_custom_collate_fn)

    # Task category 'it2i' involves image + text query to image document
    task_metadata = TaskMetadata(
        name="test_task_it2i",
        description="test it2i task",
        type="Retrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        modalities=["text", "image"],
        prompt="dummy prompt",
        dataset={"path": "dummy", "revision": "dummy"},
    )

    embeddings = wrapper.encode(
        dataloader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 128)
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_rzen_embed_wrapper_video(mock_transformers):
    """Test encoding video inputs."""
    wrapper = RzenEmbedWrapper("qihoo360/RzenEmbed", device="cpu")

    # Create dummy dataset with both text and video columns (list of frames)
    dummy_frame = Image.new("RGB", (100, 100))
    dataset = Dataset.from_dict({
        "text": ["query text 1", "query text 2"],
        "video": [[[dummy_frame, dummy_frame]], [[dummy_frame, dummy_frame]]],
    })
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=_custom_collate_fn)

    # Task category 'vt2v'
    task_metadata = TaskMetadata(
        name="test_task_vt2v",
        description="test vt2v task",
        type="Retrieval",
        category="vt2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        modalities=["text", "video"],
        prompt="dummy prompt",
        dataset={"path": "dummy", "revision": "dummy"},
    )

    embeddings = wrapper.encode(
        dataloader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 128)
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
