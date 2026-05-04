from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

UME_R1_EMBED_PROMPT = (
    "Represent the above input text, images, videos, or any combination of the three as embeddings. "
    "First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. "
    "Finally, use the <gen_emb> tag to represent the entire input."
)


class UMER1Wrapper(AbsEncoder):
    """Wrapper for UME-R1 multimodal models (uses reasoning driven generative embeddings)."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs,
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from transformers.utils.import_utils import is_flash_attn_2_available

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            attn_implementation=kwargs.get(
                "attn_implementation",
                "flash_attention_2" if is_flash_attn_2_available() else None,
            ),
            **kwargs,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        self.processor.tokenizer.padding_side = "left"

        self.gen_emb_id = self.processor.tokenizer.get_vocab().get("<gen_emb>")

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
        """Encode inputs (text and/or images) into discriminative embeddings."""
        if "video" in inputs.dataset.features:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16000,
                fps=self.fps,
                max_frames=self.max_frames,
            )

        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(inputs, desc="Encoding", disable=not show_progress_bar):
                messages = self._build_messages(batch)
                model_inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # Inference: Generation of the output
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=8192,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

                # Extract embeddings from the reasoning path
                gen_embedding = self._extract_generative_reasoning_embeddings(
                    output, model_inputs
                )
                all_embeddings.append(gen_embedding)

                # Explicitly clear large objects from GPU memory
                del output, model_inputs, messages
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

    @staticmethod
    def _build_messages(batch: dict[str, Any]) -> list[list[dict]]:
        batch_texts = batch.get("text", [])
        batch_images = batch.get("image", [])
        batch_videos = batch.get("video", [])
        batch_size = max(len(batch_texts), len(batch_images), len(batch_videos))

        messages = []
        for i in range(batch_size):
            content = []

            if batch_texts:
                content.append({"type": "text", "text": batch_texts[i]})

            if batch_images:
                image_content = batch_images[i]
                images = (
                    image_content
                    if isinstance(image_content, list)
                    else [image_content]
                )
                for img in images:
                    content.append({"type": "image", "image": img})

            if batch_videos:
                video_content = batch_videos[i]
                content.append({"type": "video", "video": video_content})

            final_text = f"<disc_emb>\n{UME_R1_EMBED_PROMPT}"
            content.append({"type": "text", "text": final_text})
            messages.append([{"role": "user", "content": content}])

        return messages

    def _get_embedding_idx(self, generated_ids_trimmed: torch.Tensor) -> list[int]:
        """Finds the step index of the embedding token for each sequence in the batch."""
        embedding_idx = []
        if self.gen_emb_id is None:
            return [-1] * generated_ids_trimmed.shape[0]

        for out_ids in generated_ids_trimmed:
            indices = (out_ids == self.gen_emb_id).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                # Add 1 because step 0 in output.hidden_states is the prefill
                embedding_idx.append(indices[-1].item() + 1)
            else:
                embedding_idx.append(-1)
        return embedding_idx

    def _extract_generative_reasoning_embeddings(
        self, output: Any, model_inputs: dict
    ) -> torch.Tensor:
        """Helper to extract embeddings from the generated output."""
        sequences = output.sequences
        input_len = model_inputs["input_ids"].shape[1]

        gen_sequences = sequences[:, input_len:]

        # Get the target steps for each sequence in the batch
        embedding_idxs = self._get_embedding_idx(gen_sequences)

        batch_reps = []
        for idx in range(sequences.shape[0]):
            target_step = embedding_idxs[idx]

            if len(gen_sequences[idx]) == 0:
                # Fallback: take the last token of the prompt if nothing generated
                emb = output.hidden_states[0][-1][idx, -1, :]
            else:
                emb = output.hidden_states[target_step][-1][idx, 0, :]

            batch_reps.append(emb)

        embeddings = torch.stack(batch_reps).cpu().to(torch.float32)
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


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
    release_date="2025-11-10",
    modalities=["image", "text", "video"],
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/DeepLearnXMU/UME-R1",
    public_training_data="https://huggingface.co/datasets/zhibinlan/UME-sft-train",
    framework=["PyTorch", "Transformers", "safetensors"],
    similarity_fn_name=ScoringFunction.COSINE,
    extra_requirements_groups=["qwen-vl"],
    use_instructions=True,
    training_datasets=None,
    citation=_UME_R1_CITATION,
)

ume_r1_2b = ModelMeta(
    name="zhibinlan/UME-R1-2B",
    revision="e7aaa253cd315be20bd15c8704ad281b4fdf82c9",
    n_parameters=2_208_985_600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=8427,
    max_tokens=32768,
    embed_dim=1536,
    reference="https://huggingface.co/zhibinlan/UME-R1-2B",
    **_UME_R1_BASE_KWARGS,
)

ume_r1_7b = ModelMeta(
    name="zhibinlan/UME-R1-7B",
    revision="b5c08e3273d979e0f22445306717c39ca8d45df0",
    n_parameters=8_291_375_616,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=31629,
    max_tokens=32768,
    embed_dim=3584,
    reference="https://huggingface.co/zhibinlan/UME-R1-7B",
    **_UME_R1_BASE_KWARGS,
)
