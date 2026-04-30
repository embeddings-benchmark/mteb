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
        from mteb._requires_package import requires_package

        requires_package(self, "qwen_vl_utils", "UME-R1", "pip install qwen_vl_utils")
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

        self.instruction_template = kwargs.pop("instruction_template", None)
        self.apply_instruction_to_passages = kwargs.pop("apply_instruction_to_passages", False)
        self.prompts_dict = kwargs.pop("prompts_dict", None)

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
        import torchvision.transforms.functional as F

        if "video" in inputs.dataset.features:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16000,
                fps=kwargs.get("fps", 2.0),
                max_frames=kwargs.get("max_frames", 64),
            )

        instruction = self.get_task_instruction(task_metadata, prompt_type)
        # print(f"Instruction: {instruction}")
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
                batch_videos = batch.get("video", [])
                batch_size = max(len(batch_texts), len(batch_images), len(batch_videos))

                messages = []
                for i in range(batch_size):
                    text_content = batch_texts[i] if batch_texts else ""
                    image_content = batch_images[i] if batch_images else None
                    video_content = batch_videos[i] if batch_videos else None

                    content = []
                    
                    if text_content:
                        content.append({"type": "text", "text": text_content})
                        
                    if image_content is not None:
                        images = image_content if isinstance(image_content, list) else [image_content]
                        for img in images:
                            content.append({"type": "image", "image": img})
                            
                    if video_content is not None:
                        if isinstance(video_content, torch.Tensor):
                            video_content = [F.to_pil_image(frame) for frame in video_content]
                        content.append({"type": "video", "video": video_content})

                    prompt = (
                        "Represent the above input text, images, videos, or any combination of the three as embeddings. "
                        "First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. "
                        "Finally, use the <gen_emb> tag to represent the entire input."
                    )
                    instr_text = instruction if instruction else ""
                    final_text = f"{instr_text}\n<disc_emb>\n{prompt}"
                    
                    content.append({"type": "text", "text": final_text})
                    # print(f"Content: {content}")
                    messages.append([{"role": "user", "content": content}])

                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    for msg in messages
                ]
                
                image_inputs, video_inputs = None, None
                if batch_images or batch_videos:
                    image_inputs, video_inputs = process_vision_info(messages)

                model_inputs = self.processor(
                    text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                ).to(self.device)

                output = self.model(**model_inputs, return_dict=True, output_hidden_states=True)
                hidden_states = output.hidden_states[-1]
                embeddings = self._extract_disc_embeddings(model_inputs["input_ids"], hidden_states, disc_emb_id)
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _extract_disc_embeddings(
        self, 
        input_ids_batch: torch.Tensor, 
        hidden_states: torch.Tensor, 
        disc_emb_id: int | None
    ) -> torch.Tensor:
        """Helper to extract embeddings for the <disc_emb> token or last token."""
        batch_reps = []
        for idx, input_ids in enumerate(input_ids_batch):
            token_idx = -1
            if disc_emb_id is not None:
                # Find the last occurrence of <disc_emb>
                indices = (input_ids == disc_emb_id).nonzero(as_tuple=True)[0]
                if indices.numel() > 0:
                    token_idx = indices[-1].item()
                else:
                    token_idx = -1
            
            batch_reps.append(hidden_states[idx, token_idx])
            
        embeddings = torch.stack(batch_reps).cpu().to(torch.float32)
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


ume_r1_task_prompts = {
    "BreakfastClassification": "What is the breakfast dish?"
}


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        return instruction[prompt_type]
    return f"{instruction}\n"


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
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        prompts_dict=ume_r1_task_prompts,
        trust_remote_code=True,
    ),
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
