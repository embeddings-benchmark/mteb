import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

from .colpali_models import (
    COLPALI_CITATION,
    COLPALI_TRAINING_DATA,
    ColPaliEngineWrapper,
)

logger = logging.getLogger(__name__)


class ColQwen2Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2,
            processor_class=ColQwen2Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


class ColQwen2_5Wrapper(ColPaliEngineWrapper):  # noqa: N801
    """Wrapper for ColQwen2.5 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else None
            )

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2_5,
            processor_class=ColQwen2_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


class ColQwen3Wrapper(AbsEncoder):
    """Wrapper for the ColQwen3 vision-language retrieval model."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | str | None = torch.bfloat16,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_package(self, "transformers", model_name, "pip install mteb[colqwen3]")
        from transformers import AutoModel, AutoProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            trust_remote_code=True,
            **kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            max_num_visual_tokens=1280,
        )

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
        if (
            "text" not in inputs.dataset.features
            and "image" not in inputs.dataset.features
        ):
            raise ValueError("No text or image features found in inputs.")
        return self.get_fused_embeddings(inputs, **kwargs)

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**encoded_inputs)
        # Avoid boolean casting of tensors when checking for custom attributes.
        embeddings = getattr(outputs, "embeddings", None)
        if embeddings is None:
            embeddings = outputs[0]
        return embeddings

    def get_fused_embeddings(
        self,
        image_texts_pairs: DataLoader[BatchedInput] | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        fusion_mode="concat",
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F
        from PIL import Image

        contains_image = "image" in image_texts_pairs.dataset.features
        contains_text = "text" in image_texts_pairs.dataset.features
        contains_both = contains_image and contains_text

        if contains_both:
            progress_desc = "Encoding images+texts"
        elif contains_image:
            progress_desc = "Encoding images"
        elif contains_text:
            progress_desc = "Encoding texts"
        else:
            raise ValueError("No text or image features found in inputs.")

        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                image_texts_pairs,
                disable=not show_progress_bar,
                desc=progress_desc,
            ):
                if contains_image:
                    imgs = [
                        F.to_pil_image(b.to(self.device))
                        if not isinstance(b, Image.Image)
                        else b
                        for b in batch["image"]
                    ]
                else:
                    imgs = None
                if contains_text:
                    texts = batch["text"]
                else:
                    texts = None
                if contains_both:
                    assert len(imgs) == len(texts), (
                        f"The number of texts and images must have the same length, got {len(imgs)} and {len(texts)}"
                    )

                inputs = self.processor(images=imgs, text=texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def similarity(self, a, b):
        return self.processor.score_multi_vector(a, b, device=self.device)


colqwen2 = ModelMeta(
    loader=ColQwen2Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colqwen2-v1.0",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-11-03",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=7200,
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2-v1.0",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

colqwen2_5 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colqwen2.5-v0.2",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6f6fcdfd1a114dfe365f529701b33d66b9349014",
    release_date="2025-01-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colqwen2.5-v0.2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

TOMORO_TRAINING_DATA = {
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
}

TOMORO_CITATION = """
@misc{huang2025tomoro_colqwen3_embed,
  title={TomoroAI/tomoro-colqwen3-embed},
  author={Xin Huang and Kye Min Tan and Albert Phelps},
  year={2025},
  url={https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b}
}
"""

colqwen3_8b = ModelMeta(
    loader=ColQwen3Wrapper,
    name="TomoroAI/tomoro-colqwen3-embed-8b",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="0b9fe28142910e209bbac15b1efe85507c27644f",
    release_date="2025-11-26",
    modalities=["image", "text"],
    n_parameters=8_000_000_000,
    memory_usage_mb=16724,
    max_tokens=262144,
    embed_dim=320,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=TOMORO_TRAINING_DATA,
    citation=TOMORO_CITATION,
)

colqwen3_4b = ModelMeta(
    loader=ColQwen3Wrapper,
    name="TomoroAI/tomoro-colqwen3-embed-4b",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6a32fb68598730bf5620fbf18d832c784235c59c",
    release_date="2025-11-26",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8466,
    max_tokens=262144,
    embed_dim=320,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=TOMORO_TRAINING_DATA,
    citation=TOMORO_CITATION,
)

colnomic_7b = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

COLNOMIC_CITATION = """
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal}
}"""

COLNOMIC_TRAINING_DATA = {"VDRMultilingual"} | COLPALI_TRAINING_DATA
COLNOMIC_LANGUAGES = [
    "deu-Latn",  # German
    "spa-Latn",  # Spanish
    "eng-Latn",  # English
    "fra-Latn",  # French
    "ita-Latn",  # Italian
]

colnomic_3b = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="nomic-ai/colnomic-embed-multimodal-3b",
    model_type=["late-interaction"],
    languages=COLNOMIC_LANGUAGES,
    revision="86627b4a9b0cade577851a70afa469084f9863a4",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
    citation=COLNOMIC_CITATION,
)

colnomic_7b = ModelMeta(
    loader=ColQwen2Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    model_type=["late-interaction"],
    languages=COLNOMIC_LANGUAGES,
    revision="09dbc9502b66605d5be56d2226019b49c9fd3293",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
    citation=COLNOMIC_CITATION,
)


EVOQWEN_TRAINING_DATA = {
    # "colpali_train_set",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
}

evoqwen25_vl_retriever_3b_v1 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="aeacaa2775f2758d82721eb1cf2f5daf1a392da9",
    release_date="2025-11-04",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=EVOQWEN_TRAINING_DATA,
)

evoqwen25_vl_retriever_7b_v1 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="8952ac6ee0e7de2e9211b165921518caf9202110",
    release_date="2025-11-04",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=EVOQWEN_TRAINING_DATA,
)
