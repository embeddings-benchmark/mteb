from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

CITATION = """@article{rzenembed2025,
  title={RzenEmbed: Multimodal Representation Learning via Vision-Language Alignment},
  author={Qihoo 360 AI Team},
  year={2025}
}"""

# Constants for Qwen2-VL smart image/video-to-patch processing
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Smart visual scaling utility ensuring compatibility with Qwen2-VL patch grids.

    Resizes images to stay within min and max pixel/token boundaries while maintaining
    aspect ratio.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / (beta * factor)) * factor
        w_bar = math.floor(width / (beta * factor)) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil((height * beta) / factor) * factor
        w_bar = math.ceil((width * beta) / factor) * factor

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logger.warning(
            f"Absolute aspect ratio {max(h_bar, w_bar) / min(h_bar, w_bar):.2f} exceeds limit {MAX_RATIO}. Adjusting aspect ratio."
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO

    return int(h_bar), int(w_bar)


def fetch_image(image: str | Image.Image | torch.Tensor | dict[str, Any], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """Robust image parser supporting PIL Images, PyTorch Tensors, local paths, HTTP/HTTPS URLs, data URIs,

    and dictionary serialization structures from datasets.
    """
    from io import BytesIO
    import base64
    import requests

    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image
        image_obj = to_pil_image(image.cpu())
    elif isinstance(image, dict) and "bytes" in image:
        image_obj = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            headers = {"User-Agent": "MTEB Evaluator 1.0"}
            response = requests.get(image, headers=headers, stream=True)
            response.raise_for_status()
            image_obj = Image.open(response.raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                image_obj = Image.open(BytesIO(base64.b64decode(base64_data)))
            else:
                raise ValueError("Unsupported base64 or data URI format.")
        else:
            image_obj = Image.open(image)
    else:
        raise TypeError(f"Unrecognized or unsupported image input format: {type(image)}")

    # Ensure RGB color space for standard vision encoders
    image_obj = image_obj.convert("RGB")
    h, w = smart_resize(image_obj.height, image_obj.width, factor=size_factor)
    return image_obj.resize((w, h))


class RzenEmbedWrapper(AbsEncoder):
    """Refactored and optimized MTEB Wrapper for qihoo360/RzenEmbed.

    Fuses and encodes text, image, and video modalities dynamically.
    """

    def __init__(
        self,
        model_name: str = "qihoo360/RzenEmbed",
        device: str | None = None,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        min_video_tokens: int = 160,
        max_video_tokens: int = 180,
        max_length: int = 2000,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = True
        self.use_instructions = True
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        # Load Qwen2-VL architecture properties
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.padding_side = "right"
        config.use_cache = False

        # Load model using optimal dtypes
        torch_dtype = torch.bfloat16 if "cuda" in str(self.device) else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # Initialize image-to-patch and text token processor
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.processor.tokenizer.padding_side = "right"

        # Initialize video-to-patch and text token processor
        min_pixels_video = min_video_tokens * 28 * 28
        max_pixels_video = max_video_tokens * 28 * 28
        self.video_processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels_video, max_pixels=max_pixels_video
        )
        self.video_processor.tokenizer.padding_side = "right"

        self.default_instruction = "You are a helpful assistant."

    def _process_images(self, images: Any) -> list[Image.Image]:
        """Maps varying input visual formats cleanly into a normalized list of PIL

        Images.
        """
        if isinstance(images, (list, tuple)):
            return [fetch_image(img) for img in images]
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return [fetch_image(frame) for frame in images]
        return [fetch_image(images)]

    def _prepare_sample(
        self, text: str | None, visual: Any, instruction: str | None = None
    ) -> tuple[str, list[Image.Image] | None]:
        """Assembles prompt strings and vision padding tokens matching Qwen2-VL's prompt

        structure.
        """
        input_str = ""
        processed_images = None

        if visual is not None:
            # Video frames can be passed inside nested lists or sequential iterables
            if isinstance(visual, list) and len(visual) > 0 and isinstance(visual[0], list):
                processed_images = self._process_images(visual[0])
            else:
                processed_images = self._process_images(visual)

            input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_images)

        # Prepend task prompt guidelines matching Qihoo 360 official inference
        if instruction:
            input_str += instruction

        if text is not None:
            input_str += text

        # Standard Qwen2-VL chat format
        prompt = (
            f"<|im_start|>system\n{self.default_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{input_str}<|im_end|>\n"
            f"<|im_start|>assistant\n<|endoftext|>"
        )
        return prompt, processed_images

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Safe retrieval of task instructions to guard against core/task-side AttributeError exceptions."""
        try:
            return super().get_task_instruction(task_metadata, prompt_type)
        except AttributeError:
            if task_metadata.prompt:
                if isinstance(task_metadata.prompt, str):
                    return task_metadata.prompt
                if isinstance(task_metadata.prompt, dict) and prompt_type:
                    return task_metadata.prompt.get(prompt_type.value, "")
            return ""

    @torch.no_grad()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Main entry point orchestrating text, image, video and joint multi-modality

        encoding.
        """
        import numpy as np

        features = inputs.dataset.features
        has_video = "video" in features

        # Register standardized VideoCollator when handling video streams
        if has_video:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        # Gather MTEB task instructions
        instruction = ""
        if self.use_instructions:
            instruction = self.get_task_instruction(task_metadata, prompt_type)

        all_embeddings = []

        # Iterate over batched inputs yielded by MTEB DataLoader
        for batch in tqdm(inputs, desc="RzenEmbed Processing"):
            texts = batch.get("text", None)
            images = batch.get("image", None)
            videos = batch.get("video", None)
            visuals = videos if has_video else images

            batch_size = len(texts) if texts is not None else len(visuals)
            input_texts = []
            batch_images = []

            for i in range(batch_size):
                t = texts[i] if texts is not None else None
                v = visuals[i] if visuals is not None else None

                prompt, processed_visuals = self._prepare_sample(t, v, instruction=instruction)
                input_texts.append(prompt)
                if processed_visuals is not None:
                    batch_images.extend(processed_visuals)

            if len(batch_images) == 0:
                batch_images = None

            # Tokenize & encode without token-level truncation to prevent PyTorch shape mismatch crashes
            processor_to_use = self.video_processor if has_video else self.processor
            inputs_tokenized = processor_to_use(
                text=input_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )

            # Relocate tensors to active hardware
            inputs_tokenized = {k: v.to(self.device) for k, v in inputs_tokenized.items()}

            inputs_embeds = self.model.model.language_model.embed_tokens(inputs_tokenized["input_ids"])
            pixel_values = inputs_tokenized.get("pixel_values", None)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.model.visual.get_dtype())
                image_embeds = self.model.model.visual(
                    pixel_values, grid_thw=inputs_tokenized["image_grid_thw"]
                )
                
                # Retrieve the raw tensor from BaseModelOutputWithPooling if needed
                if not isinstance(image_embeds, torch.Tensor):
                    if hasattr(image_embeds, "last_hidden_state") and image_embeds.last_hidden_state is not None:
                        image_embeds = image_embeds.last_hidden_state
                    elif hasattr(image_embeds, "image_embeds") and image_embeds.image_embeds is not None:
                        image_embeds = image_embeds.image_embeds
                    else:
                        image_embeds = image_embeds[0]

                # Project visual embeddings using the visual merger to map to language model dimensions
                image_embeds = self.model.model.visual.merger(image_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device)

                image_mask = (
                    inputs_tokenized["input_ids"] == self.model.config.image_token_id
                )
                inputs_embeds[image_mask] = image_embeds

            outputs = self.model.model(
                input_ids=None,
                position_ids=inputs_tokenized.get("position_ids", None),
                attention_mask=inputs_tokenized.get("attention_mask", None),
                inputs_embeds=inputs_embeds,
            )

            # Perform pooling to retrieve the last active token representation
            attention_mask = inputs_tokenized["attention_mask"]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=outputs.last_hidden_state.device),
                sequence_lengths,
            ]

            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().to(torch.float32))

        return np.concatenate([emb.numpy() for emb in all_embeddings], axis=0)


rzen_embed = ModelMeta(
    loader=RzenEmbedWrapper,
    name="qihoo360/RzenEmbed",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    revision="ea95c339008f9639420fcb33ba2ba606ea94be78",
    release_date="2025-11-06",
    modalities=["image", "video", "text"],
    n_parameters=8_291_375_616,
    memory_usage_mb=16584,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    max_tokens=32768,
    reference="https://huggingface.co/qihoo360/RzenEmbed",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
