from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import mteb
from mteb.mocks.mock_tasks.retrieval import MockMultiChoiceTask
from mteb.models.model_implementations.rzen_embed_model import RzenEmbedWrapper

logging.basicConfig(level=logging.INFO)


# We patch transformers before initializing and running the evaluation
with patch("transformers.AutoConfig.from_pretrained") as mock_config, \
     patch("transformers.AutoProcessor.from_pretrained") as mock_processor, \
     patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model:

    import torch

    # Setup mocks
    config = MagicMock()
    mock_config.return_value = config

    processor = MagicMock()
    video_processor = MagicMock()

    processor.tokenizer = MagicMock()
    processor.tokenizer.padding_side = "right"
    video_processor.tokenizer = MagicMock()
    video_processor.tokenizer.padding_side = "right"

    def mock_processor_call(text, images=None, **kwargs):
        batch_size = len(text) if text is not None else len(images)
        seq_len = 5
        features = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }
        if images is not None:
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

    # Setup mock model directly using standard MagicMock assignments
    model_instance = MagicMock()
    model_instance.to.return_value = model_instance
    model_instance.config.image_token_id = 151655
    
    mock_language_model = MagicMock()
    mock_language_model.embed_tokens.return_value = torch.zeros((2, 5, 128), dtype=torch.float32)
    
    mock_visual = MagicMock()
    mock_visual.get_dtype.return_value = torch.float32
    mock_visual.return_value = torch.zeros((2, 128), dtype=torch.float32)
    
    mock_model_inner = MagicMock()
    mock_model_inner.language_model = mock_language_model
    mock_model_inner.visual = mock_visual
    
    # mock outputs forward pass
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = torch.ones((2, 5, 128), dtype=torch.float32)
    mock_model_inner.return_value = mock_outputs
    
    model_instance.model = mock_model_inner
    mock_model.return_value = model_instance

    from mteb.models.model_implementations.rzen_embed_model import rzen_embed

    # 1. Initialize RzenEmbedWrapper (which loads the mocked dependencies)
    rzen_model = RzenEmbedWrapper("qihoo360/RzenEmbed", device="cpu")
    rzen_model.mteb_model_meta = rzen_embed

    # 2. Instantiate our it2i target task: MockMultiChoiceTask
    task = MockMultiChoiceTask()

    # 3. Create MTEB runner and execute evaluation on the task
    evaluation = mteb.MTEB(tasks=[task])
    results = evaluation.run(rzen_model, output_folder="results/mock_rzen")

    print("\n" + "=" * 50)
    print("\nMTEB Evaluation on MockMultiChoiceTask (it2i) Completed Successfully!")
    print("Results:")
    for result in results:
        print(f"Task: {result.task_name}")
        for split, split_results in result.scores.items():
            print(f"  Split: {split}")
            for metric, value in split_results[0].items():
                print(f"    {metric}: {value}")
    print("=" * 50)
