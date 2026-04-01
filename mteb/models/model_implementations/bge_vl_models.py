from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

MEGAPAIRS_CITATION = """@article{zhou2024megapairs,
  title={MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval},
  author={Zhou, Junjie and Liu, Zheng and Liu, Ze and Xiao, Shitao and Wang, Yueze and Zhao, Bo and Zhang, Chen Jason and Lian, Defu and Xiong, Yongping},
  journal={arXiv preprint arXiv:2412.14475},
  year={2024}
}"""

MLLM_TASK_INSTRUCTION = (
    "Retrieve the target image that best meets the combined criteria by using both "
    "the provided image and the image retrieval instructions: "
)

BGE_VL_TRAINING_DATASETS = {"MegaPairs"}


class BGEVLModel(AbsEncoder):
    """Wrapper for BGE-VL CLIP and MLLM model families."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel

        requires_image_dependencies()

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            **kwargs,
        ).eval()
        self.model = self.model.to(self.device)

        if hasattr(self.model, "set_processor"):
            self.model.set_processor(model_name)

    @staticmethod
    def _prepare_images(images: list) -> list[Image.Image]:
        import torchvision.transforms.functional as tv_functional
        from PIL import Image

        prepared: list[Image.Image] = []
        for image in images:
            if isinstance(image, Image.Image):
                prepared.append(image)
            else:
                prepared.append(tv_functional.to_pil_image(image.cpu()))
        return prepared

    @staticmethod
    def _pool_last_token(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_state[:, -1, :]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        row_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row_idx, sequence_lengths]

    def _encode_with_clip_api(
        self,
        *,
        text: list[str] | None,
        images: list[Image.Image] | None,
    ) -> torch.Tensor:
        if images is not None:
            converted_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    bytes_io = io.BytesIO()
                    img.save(bytes_io, format="PNG")
                    bytes_io.seek(0)
                    converted_images.append(bytes_io)
                else:
                    converted_images.append(img)
            images = converted_images

        return self.model.encode(images=images, text=text)

    def _encode_with_mllm_api(
        self,
        *,
        text: list[str] | None,
        images: list[Image.Image] | None,
        task_instruction: str,
    ) -> torch.Tensor:
        import io

        from PIL import Image

        # Convert PIL Images to BytesIO objects so Image.open() works in data_process
        if images is not None:
            converted_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    bytes_io = io.BytesIO()
                    img.save(bytes_io, format="PNG")
                    bytes_io.seek(0)
                    converted_images.append(bytes_io)
                else:
                    converted_images.append(img)
            images = converted_images

        if images is not None and text is not None:
            query_inputs = self.model.data_process(
                text=text,
                images=images,
                q_or_c="q",
                task_instruction=task_instruction,
            )
        elif images is not None:
            query_inputs = self.model.data_process(images=images, q_or_c="c")
        elif text is not None:
            query_inputs = self.model.data_process(
                text=text,
                q_or_c="q",
                task_instruction=task_instruction,
            )
        else:
            raise ValueError("Neither text nor image found in the input batch.")

        query_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in query_inputs.items()
        }

        outputs = self.model(**query_inputs, output_hidden_states=True)

        # Most MLLM BGE-VL checkpoints return [batch, seq, hidden] directly,
        # while some return a model output object with hidden states.
        if isinstance(outputs, torch.Tensor):
            hidden = outputs
            attention_mask = query_inputs.get("attention_mask")
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            hidden = outputs.last_hidden_state
            attention_mask = query_inputs.get("attention_mask")
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
            attention_mask = query_inputs.get("attention_mask")
        else:
            raise RuntimeError("Unexpected output format from BGE-VL MLLM model.")

        embeddings = self._pool_last_token(hidden, attention_mask)
        return F.normalize(embeddings, dim=-1)

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
        contains_text = "text" in inputs.dataset.features
        contains_image = "image" in inputs.dataset.features

        task_instruction = self.get_instruction(task_metadata, prompt_type)
        if prompt_type == PromptType.document:
            task_instruction = None
        if not task_instruction:
            task_instruction = MLLM_TASK_INSTRUCTION

        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
                batch_size = len(batch["text"] if contains_text else batch["image"])
                texts = list(batch["text"]) if contains_text else None
                images = batch["image"] if contains_image else None

                if texts is not None and len(texts) != batch_size:
                    raise ValueError("Mismatch between batch size and text field size.")

                if hasattr(self.model, "encode"):
                    embeddings = self._encode_with_clip_api(text=texts, images=images)
                else:
                    embeddings = self._encode_with_mllm_api(
                        text=texts,
                        images=images,
                        task_instruction=task_instruction,
                    )

                all_embeddings.append(embeddings.detach().cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0)


bge_vl_base = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-base",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="1aba80d3faf358474cfdd26baccda8b4f7ff2f35",
    release_date="2025-02-25",
    modalities=["image", "text"],
    n_parameters=149_620_737,
    n_embedding_parameters=25_296_896,
    memory_usage_mb=285,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)

bge_vl_large = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="41c4fb301c5642372b3290ea8027cb9149c4f7bc",
    release_date="2025-02-25",
    modalities=["image", "text"],
    n_parameters=427_616_513,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=816,
    max_tokens=77,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)

bge_vl_mllm_s1 = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-MLLM-S1",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="27146dfb4c33a56732244f487844ec9f7963eba3",
    release_date="2025-03-04",
    modalities=["image", "text"],
    n_parameters=7_566_264_320,
    n_embedding_parameters=131092480,
    memory_usage_mb=14432,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-MLLM-S1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)

bge_vl_mllm_s2 = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-MLLM-S2",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="6a96116cb5d7c95e123fe484c0b60cba4445175a",
    release_date="2025-03-04",
    modalities=["image", "text"],
    n_parameters=7_566_264_320,
    n_embedding_parameters=131092480,
    memory_usage_mb=14432,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-MLLM-S2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)

bge_vl_v1_5_zs = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-v1.5-zs",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="4ba848713c03880710c1dffd54cb179078b3fd4b",
    release_date="2025-05-15",
    modalities=["image", "text"],
    n_parameters=7_566_747_648,
    n_embedding_parameters=131334144,
    memory_usage_mb=14432,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-v1.5-zs",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)

bge_vl_v1_5_mmeb = ModelMeta(
    loader=BGEVLModel,
    name="BAAI/BGE-VL-v1.5-mmeb",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="a26da62e8dd47e47e78c3b7a1d6877053f409a1d",
    release_date="2025-05-16",
    modalities=["image", "text"],
    n_parameters=7_566_747_648,
    n_embedding_parameters=131334144,
    memory_usage_mb=14432,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/VectorSpaceLab/MegaPairs",
    public_training_data="https://huggingface.co/datasets/JUNJIE99/MegaPairs",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/BAAI/BGE-VL-v1.5-mmeb",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=BGE_VL_TRAINING_DATASETS,
    citation=MEGAPAIRS_CITATION,
)
