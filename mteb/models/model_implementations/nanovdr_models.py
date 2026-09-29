from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from tqdm.autonotebook import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from transformers.cache_utils import Cache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig

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


# Vendored from mteb/models/model_implementations/qwen3_vl_embedding_models.py,
# where it was removed in beee2102 (#4699) when Qwen3VLEmbeddingWrapper moved to
# InstructSentenceTransformerModel. This wrapper is its only remaining consumer,
# so it lives here rather than being re-added to that module's public surface.
def _build_qwen3_vl_for_embedding_class() -> type:
    """Lazily construct the custom Qwen3VLForEmbedding model class.

    This class mirrors the official ``Qwen3VLForEmbedding`` from the model
    repository scripts.  It wraps ``Qwen3VLModel`` (without the LM head)
    so that we can extract ``last_hidden_state`` directly, which is the
    behaviour intended by the model authors.
    """
    from dataclasses import dataclass

    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    @dataclass
    class Qwen3VLForEmbeddingOutput(ModelOutput):
        last_hidden_state: torch.FloatTensor | None = None
        attention_mask: torch.Tensor | None = None

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping: ClassVar[dict] = {}
        accepts_loss_kwargs = False

        def __init__(self, config: Qwen3VLConfig):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value: torch.nn.Module) -> None:
            self.model.set_input_embeddings(value)

        def get_video_features(
            self,
            pixel_values_videos: torch.FloatTensor,
            video_grid_thw: torch.LongTensor | None = None,
        ) -> torch.Tensor:
            return self.model.get_video_features(pixel_values_videos, video_grid_thw)

        def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            image_grid_thw: torch.LongTensor | None = None,
        ) -> torch.Tensor:
            return self.model.get_image_features(pixel_values, image_grid_thw)

        @property
        def language_model(self) -> torch.nn.Module:
            return self.model.language_model

        @property
        def visual(self) -> torch.nn.Module:
            return self.model.visual

        def forward(  # noqa: PLR0913, PLR0917
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs: Any,
        ) -> tuple | Qwen3VLForEmbeddingOutput:
            # Setting to None enables image + text embeddings mode
            # More info: https://github.com/embeddings-benchmark/mteb/pull/4198/changes#r2899945802
            self.model.rope_deltas = None

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )
            return Qwen3VLForEmbeddingOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

    return Qwen3VLForEmbedding


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
        import torch

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

        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

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
        # convert_to_tensor=True, not convert_to_numpy=False: the latter leaves
        # convert_to_tensor at its default and sentence-transformers then returns
        # a list[Tensor], which mteb's _convert_to_tensor cannot stack. .cpu()
        # matches _encode_documents -- cos_sim does no device harmonisation, so
        # mixing a CUDA query matrix with a CPU corpus matrix would fail.
        return self.query_model.encode(
            all_texts,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        ).cpu()

    def _encode_documents(  # noqa: PLR0914
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
    ) -> Array:
        """Encode document page images using the Qwen3-VL teacher."""
        import unicodedata

        import torch
        import torch.nn.functional as F
        from qwen_vl_utils.vision_process import process_vision_info

        self._load_teacher()

        import torchvision.transforms.functional as tv_functional
        from PIL import Image

        instruction = QUERY_INSTRUCTION.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction = instruction + "."  # noqa: PLR6104

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
                    videos, video_metadata = zip(*video_inputs, strict=True)
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
            prompt_type in (PromptType.query, None)  # noqa: PLR6201
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
    extra_requirements_groups=["qwen-vl"],
)
