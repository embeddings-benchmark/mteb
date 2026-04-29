from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

class UMER1Wrapper(AbsEncoder):
    """Wrapper for UME-R1 multimodal models (uses discriminative embeddings for efficient evaluation)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from transformers.utils.import_utils import is_flash_attn_2_available

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_name = model_name

        attn_impl = kwargs.pop(
            "attn_implementation",
            "flash_attention_2" if is_flash_attn_2_available() else None,
        )

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            **kwargs,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

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
        """Encode inputs (text and/or images) into discriminative embeddings."""
        from qwen_vl_utils import process_vision_info

        all_embeddings: list[torch.Tensor] = []
        disc_emb_id = self.processor.tokenizer.get_vocab().get("<disc_emb>")
        if disc_emb_id is None:
            logger.warning(
                "<disc_emb> token not found in tokenizer. Extracting the last token instead."
            )

        with torch.no_grad():
            for batch in tqdm(inputs, desc="Encoding"):
                batch_texts = batch.get("text", [])
                batch_images = batch.get("image", [])

                messages = []
                for i in range(max(len(batch_texts), len(batch_images))):
                    text_content = batch_texts[i] if batch_texts else ""
                    image_content = batch_images[i] if batch_images else None

                    content = []
                    if image_content is not None:
                        content.append(
                            {
                                "type": "image",
                                "image": image_content,
                            }
                        )

                    # Determine text to add
                    query_prefix = "Query: " if prompt_type == PromptType.query else ""
                    if text_content:
                        text_part = f"{query_prefix}{text_content}"
                    else:
                        text_part = "Represent the given image."

                    text_part += "\n<disc_emb>\n"

                    content.append({"type": "text", "text": text_part})
                    messages.append([{"role": "user", "content": content}])

                # Prepare inputs
                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    for msg in messages
                ]

                image_inputs = None
                video_inputs = None
                if batch_images:
                    image_inputs, video_inputs = process_vision_info(messages)

                model_inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # Get embeddings
                output = self.model(
                    **model_inputs, return_dict=True, output_hidden_states=True
                )
                hidden_states = output.hidden_states[-1]

                # Extract embeddings for each item in the batch
                batch_reps = []
                for idx, input_ids in enumerate(model_inputs["input_ids"]):
                    token_idx = -1
                    if disc_emb_id is not None:
                        for j in range(len(input_ids) - 1, -1, -1):
                            if input_ids[j] == disc_emb_id:
                                token_idx = j
                                break

                    rep = hidden_states[idx, token_idx]
                    batch_reps.append(rep)

                embeddings = torch.stack(batch_reps)
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

_UME_R1_CITATION = """@article{lan2025ume,
  title={UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings},
  author={Lan, Zhibin and Niu, Liqiang and Meng, Fandong and Zhou, Jie and Su, Jinsong},
  journal={arXiv preprint arXiv:2511.00405},
  year={2025}
}"""

_UME_R1_BASE_KWARGS = dict(
    loader=UMER1Wrapper,
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-11-10",
    modalities=["image", "text", "video"],
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/DeepLearnXMU/UME-R1",
    public_training_data="https://huggingface.co/datasets/zhibinlan/UME-sft-train",
    framework=["PyTorch", "Transformers", "safetensors"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    citation=_UME_R1_CITATION,
)

ume_r1_2b = ModelMeta(
    name="zhibinlan/UME-R1-2B",
    revision="e7aaa253cd315be20bd15c8704ad281b4fdf82c9",
    n_parameters=2_208_985_600,
    n_embedding_parameters=None,
    memory_usage_mb=4500,
    max_tokens=32768,
    embed_dim=1536,
    reference="https://huggingface.co/zhibinlan/UME-R1-2B",
    **_UME_R1_BASE_KWARGS,
)

ume_r1_7b = ModelMeta(
    name="zhibinlan/UME-R1-7B",
    revision="b5c08e3273d979e0f22445306717c39ca8d45df0",
    n_parameters=8_291_375_616,
    n_embedding_parameters=None,
    memory_usage_mb=16000,
    max_tokens=32768,
    embed_dim=3584,
    reference="https://huggingface.co/zhibinlan/UME-R1-7B",
    **_UME_R1_BASE_KWARGS,
)
