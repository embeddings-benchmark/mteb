from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

NANOVDR_CITATION = """@article{nanovdr2026,
  title={NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval},
  author={Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal={arXiv preprint arXiv:2603.12824},
  year={2026}
}"""

QUERY_INSTRUCTION = "Find a document image that matches the given query."


class NanoVDRWrapper(AbsEncoder):
    """Asymmetric retrieval wrapper for NanoVDR.

    Routes queries to a lightweight text-only SentenceTransformer student
    and documents to the frozen Qwen3-VL-Embedding-2B VLM teacher.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ):
        requires_package(
            self,
            "sentence_transformers",
            model_name,
            "pip install sentence-transformers",
        )

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Query encoder: lightweight text-only student
        from sentence_transformers import SentenceTransformer

        self.query_model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device,
        )

        # Document encoder: frozen VLM teacher (lazy-loaded on first use)
        self._doc_model = None
        self._doc_processor = None

    def _load_teacher(self) -> None:
        """Lazily load the Qwen3-VL-Embedding-2B teacher for document encoding."""
        if self._doc_model is not None:
            return

        requires_image_dependencies()
        requires_package(
            self,
            "transformers",
            "Qwen/Qwen3-VL-Embedding-2B",
            "pip install 'mteb[qwen-vl]'",
        )
        requires_package(
            self,
            "qwen_vl_utils",
            "Qwen/Qwen3-VL-Embedding-2B",
            "pip install 'mteb[qwen-vl]'",
        )

        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        from mteb.models.model_implementations.qwen3_vl_embedding_models import (
            _build_qwen3_vl_for_embedding_class,
        )

        qwen3_vl_cls = _build_qwen3_vl_for_embedding_class()
        self._doc_model = qwen3_vl_cls.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-2B",
        ).to(self.device)
        self._doc_model.eval()
        self._doc_processor = Qwen3VLProcessor.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-2B",
            padding_side="right",
        )

    def _encode_queries(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
    ) -> Array:
        all_texts = [text for batch in inputs for text in batch["text"]]
        return self.query_model.encode(
            all_texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=False,
        )

    def _encode_documents(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
    ) -> Array:
        """Encode document page images using the Qwen3-VL teacher."""
        import unicodedata

        from qwen_vl_utils.vision_process import process_vision_info

        self._load_teacher()

        import torchvision.transforms.functional as tv_functional
        from PIL import Image

        instruction = QUERY_INSTRUCTION.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction = instruction + "."

        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                inputs, disable=not show_progress_bar, desc="Encoding docs"
            ):
                contains_image = "image" in batch and batch["image"] is not None
                contains_text = "text" in batch

                batch_size = len(batch["image"] if contains_image else batch["text"])

                conversations = []
                for i in range(batch_size):
                    content: list[dict[str, Any]] = []
                    if contains_image:
                        img = batch["image"][i]
                        if isinstance(img, Image.Image):
                            pil_img = img
                        else:
                            pil_img = tv_functional.to_pil_image(img.cpu())
                        content.append(
                            {
                                "type": "image",
                                "image": pil_img,
                                "min_pixels": 4 * 32 * 32,
                                "max_pixels": 1800 * 32 * 32,
                            }
                        )
                    if contains_text and not contains_image:
                        text = batch["text"][i] if batch["text"][i] else "NULL"
                        content.append({"type": "text", "text": text})

                    conversations.append(
                        [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Represent the user's input.",
                                    }
                                ],
                            },
                            {"role": "user", "content": content},
                        ]
                    )

                text = self._doc_processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                try:
                    images, video_inputs, video_kwargs = process_vision_info(
                        conversations,
                        image_patch_size=16,
                        return_video_metadata=True,
                        return_video_kwargs=True,
                    )
                except Exception:
                    images, video_inputs, video_kwargs = (
                        None,
                        None,
                        {"do_sample_frames": False},
                    )

                videos, video_metadata = None, None
                if video_inputs is not None:
                    videos, video_metadata = zip(*video_inputs)
                    videos, video_metadata = list(videos), list(video_metadata)

                processed = self._doc_processor(
                    text=text,
                    images=images,
                    videos=videos,
                    video_metadata=video_metadata,
                    truncation=True,
                    max_length=8192,
                    padding=True,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
                processed = {k: v.to(self.device) for k, v in processed.items()}

                outputs = self._doc_model(**processed)

                # Last-token pooling
                attn = processed["attention_mask"]
                flipped = attn.flip(dims=[1])
                last_pos = flipped.argmax(dim=1)
                col = attn.shape[1] - last_pos - 1
                row = torch.arange(
                    outputs.last_hidden_state.shape[0], device=self.device
                )
                embeddings = outputs.last_hidden_state[row, col]
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

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
        from mteb.types import PromptType

        # Validate: reject tasks where queries contain images.
        # NanoVDR only supports text-query → image-document retrieval.
        if (
            prompt_type in (PromptType.query, None)
            and "image" in inputs.dataset.features
        ):
            raise ValueError(
                f"NanoVDR only supports text queries, but task "
                f"'{task_metadata.name}' provides image inputs for queries. "
                f"NanoVDR is a text-query → image-document retrieval model "
                f"and does not support image-query or image-classification tasks."
            )

        if prompt_type == PromptType.document:
            return self._encode_documents(inputs, show_progress_bar=show_progress_bar)
        else:
            # Use the lightweight student for queries and all non-retrieval tasks
            return self._encode_queries(inputs, show_progress_bar=show_progress_bar)


nanovdr_s_multi = ModelMeta(
    loader=NanoVDRWrapper,
    name="nanovdr/NanoVDR-S-Multi",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "fra-Latn", "spa-Latn", "ita-Latn", "por-Latn"],
    open_weights=True,
    revision="b21574d7772ca26e22525543a2a6bf7081a95d8f",
    release_date="2026-02-26",
    modalities=["text", "image"],
    n_parameters=69_000_000,
    n_embedding_parameters=1_572_864,
    memory_usage_mb=282,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/nanovdr/NanoVDR-S-Multi",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/nanovdr/NanoVDR-Train",
    training_datasets={
        "VidoreTabfquadRetrieval",
        "VidoreDocVQARetrieval",
        "VidoreInfoVQARetrieval",
        "VidoreArxivQARetrieval",
    },
    citation=NANOVDR_CITATION,
)
