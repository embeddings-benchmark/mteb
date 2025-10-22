import logging
import math
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

GME_CITATION = """@misc{zhang2024gme,
      title={GME: Improving Universal Multimodal Retrieval by Multimodal LLMs},
      author={Zhang, Xin and Zhang, Yanzhao and Xie, Wen and Li, Mingxin and Dai, Ziqi and Long, Dingkun and Xie, Pengjun and Zhang, Meishan and Li, Wenjie and Zhang, Min},
      year={2024},
      eprint={2412.16855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={http://arxiv.org/abs/2412.16855}
}"""


class Encoder(torch.nn.Module):
    def __init__(
        self,
        base,
        processor,
        max_length=1800,
        normalize=True,
    ) -> None:
        super().__init__()
        self.base = base
        self.processor = processor
        self.max_length = max_length
        self.normalize = normalize
        self.processor.tokenizer.padding_side = "right"
        self.default_instruction = "You are a helpful assistant."

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        # pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        # video_grid_thw: torch.LongTensor | None = None,
        pooling_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            # if pixel_values_videos is not None:
            #     pixel_values_videos = pixel_values_videos.type(self.base.visual.get_dtype())
            #     video_embeds = self.base.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
            #     video_mask = input_ids == self.base.config.video_token_id
            #     inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.base.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]  # TODO
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=outputs.last_hidden_state.device),
                sequence_lengths,
            ]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(
        self,
        texts: list[str],
        images: list[Image.Image],
        device,
        instruction=None,
        **kwargs,
    ):
        instruction = instruction or self.default_instruction
        # Inputs must be batched
        input_texts, input_images = [], []
        for t, i in zip(texts, images):
            input_str = ""
            if i is None:
                input_images = None  # All examples in the same batch are consistent
            else:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                i = fetch_image(i)
                input_images.append(i)
            if t is not None:
                input_str += t
            msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}  # TODO
        embeddings = self.forward(**inputs)
        return embeddings


class GmeQwen2VL(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        model_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_image_tokens=4,
        max_image_tokens=1280,
        max_length=1800,
        **kwargs,
    ) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        model_name = model_path or model_name
        base = AutoModelForVision2Seq.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.float16, **kwargs
        )
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs,
        )
        self.model = Encoder(base, processor, max_length=max_length)
        self.model.eval()
        self.device = device
        self.sep = " "

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = self.get_instruction(task_metadata, prompt_type)
        if prompt_type == PromptType.document:
            instruction = None
        elif instruction is None:
            instruction = self.get_instruction(task_metadata, prompt_type)
            # NOTE: copied from the old get_gme_instruction function.
            if isinstance(instruction, str) and instruction[-1] != ".":
                instruction += "."
        self.model = self.model.to(self.device)
        all_embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Fused Encoding"):
            batch_size = len(batch["text"]) or len(batch["image"])
            if "text" in batch:
                text_batch = batch["text"]
            else:
                text_batch = [None] * batch_size

            if "image" in batch:
                img_batch = batch["image"]
            else:
                img_batch = [None] * batch_size

            inputs = dict(
                texts=text_batch, images=img_batch, instruction=instruction, **kwargs
            )
            with torch.inference_mode():
                embeddings = self.model.embed(**inputs, device=self.device)
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


# Copied from qwen_vl_utils.vision_process.py

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logger.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(
    image: str | Image.Image, size_factor: int = IMAGE_FACTOR
) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        import requests

        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        import base64
        from io import BytesIO

        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = image_obj.convert("RGB")
    # resize
    # if "resized_height" in ele and "resized_width" in ele:
    #     resized_height, resized_width = smart_resize(
    #         ele["resized_height"],
    #         ele["resized_width"],
    #         factor=size_factor,
    #     )
    # else:
    width, height = image.size
    # min_pixels = ele.get("min_pixels", MIN_PIXELS)
    # max_pixels = ele.get("max_pixels", MAX_PIXELS)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))

    return image


training_data = {
    "MSMARCO",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQAHardNegatives",
    # TriviaQA (Joshi et al., 2017),
    # SQuAD (Rajpurkar et al., 2016),
    "FEVER",
    # AllNLI for SimCSE (Gao et al., 2021), selecting a total of 1 million entries.
    # ImageNet (Deng et al., 2009)
    # LAION (Schuhmann et al., 2022),
    # mscoco (Lin et al., 2014),
    # Docmatix (Laurenc¸on et al., 2024)
    # synthetic data
    # M-BEIR (Wei et al., 2024)
}


gme_qwen2vl_2b = ModelMeta(
    loader=GmeQwen2VL,
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    languages=["eng-Latn", "cmn-Hans"],
    open_weights=True,
    revision="ce765ae71b8cdb208203cd8fb64a170b1b84293a",
    release_date="2024-12-24",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=8427,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=GME_CITATION,
)

gme_qwen2vl_7b = ModelMeta(
    loader=GmeQwen2VL,
    name="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    languages=["eng-Latn", "cmn-Hans"],
    open_weights=True,
    revision="477027a6480f8630363be77751f169cc3434b673",
    release_date="2024-12-24",
    modalities=["image", "text"],
    n_parameters=8_290_000_000,
    memory_usage_mb=31629,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=GME_CITATION,
)
