from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

logger = logging.getLogger(__name__)

VIRTUE_CITATION = """@article{wang2025virtue,
  title={VIRTUE: Visual-Interactive Text-Image Universal Embedder},
  author={Wang, Wei-Yao and Tateishi, Kazuya and Wu, Qiyu and Takahashi, Shusuke and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2510.00523},
  year={2025}
}"""

_VIRTUE_IMAGE_TOKEN = "<|image_pad|>"  # noqa: S105


class VirtueWrapper(AbsEncoder):
    """Wrapper for the VIRTUE universal embedder (https://github.com/sony/virtue).

    VIRTUE is built on Qwen2-VL and additionally supports visual prompts (point,
    bounding box, mask) via a SAM2 module. MTEB tasks do not provide visual
    prompts, so the model is used as a standard Qwen2-VL embedder with last-token
    pooling and L2 normalization, matching the no-visual-prompt path of the
    reference implementation.
    """

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        **kwargs,
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        self.processor.padding_side = "left"

        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, revision=revision, torch_dtype=torch_dtype, **kwargs
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pooling(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        batch_size = last_hidden_state.shape[0]
        if left_padding:
            reps = last_hidden_state[torch.arange(batch_size), -1, :]
        else:
            eos_indices = attention_mask.sum(dim=1) - 1
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), eos_indices
            ]
        return torch.nn.functional.normalize(reps, p=2, dim=-1)

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features

        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            batch_size = len(next(iter(batch.values())))

            prompts: list[str] = []
            batch_images: list | None = [] if has_image else None

            for i in range(batch_size):
                text = batch["text"][i] if has_text else None
                if has_image:
                    if text:
                        prompt = (
                            f"{_VIRTUE_IMAGE_TOKEN}\nRepresent the given image with "
                            f"the following question: {text}"
                        )
                    else:
                        prompt = f"{_VIRTUE_IMAGE_TOKEN}\nRepresent the given image."
                    batch_images.append(batch["image"][i])
                else:
                    prompt = text if text else ""
                prompts.append(prompt)

            proc_inputs = self.processor(
                text=prompts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )
            proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}

            # Call the inner model to obtain hidden states directly and avoid the
            # memory-heavy `lm_head` projection over the full vocabulary.
            output = self.model.model(**proc_inputs, return_dict=True)
            hidden_states = output.last_hidden_state
            embs = self._pooling(hidden_states, proc_inputs["attention_mask"])
            all_embeddings.append(embs.cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0)


virtue_training_datasets = set(
    # MMEB-train
    # SCaR-train
)

virtue_2b_scar = ModelMeta(
    loader=VirtueWrapper,
    name="Sony/VIRTUE-2B-SCaR",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="e98d03976ccbc7738c5ca14fab98d442a06c0530",
    release_date="2026-02-03",
    modalities=["image", "text"],
    n_parameters=2208985600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=8427,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Sony/VIRTUE-2B-SCaR",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/sony/virtue",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    training_datasets=virtue_training_datasets,
    adapted_from="Qwen/Qwen2-VL-2B-Instruct",
    citation=VIRTUE_CITATION,
)

virtue_7b_scar = ModelMeta(
    loader=VirtueWrapper,
    name="Sony/VIRTUE-7B-SCaR",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="e5280a7cb6a08961782b04a9753aa31351656b66",
    release_date="2026-02-03",
    modalities=["image", "text"],
    n_parameters=7746378240,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=31629,
    embed_dim=3584,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Sony/VIRTUE-7B-SCaR",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/sony/virtue",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    training_datasets=virtue_training_datasets,
    adapted_from="Qwen/Qwen2-VL-7B-Instruct",
    citation=VIRTUE_CITATION,
)
