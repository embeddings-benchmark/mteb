from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from mteb._requires_package import requires_image_dependencies
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class OpsColQwen3Wrapper(AbsEncoder):
    """Wrapper for OpsColQwen3 model."""

    def __init__(
        self,
        model_name: str = "OpenSearch-AI/Ops-Colqwen3-4B",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else None
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mdl = AutoModel.from_pretrained(
            model_name,
            device_map=self.device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            revision=revision,
            **kwargs,
        )
        self.mdl.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
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
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image inputs found")

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images: DataLoader,
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                # batch may be list of tensors or PIL
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts: DataLoader,
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                batch_texts = batch["text"]
                inputs = self.processor.process_queries(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.processor.score_multi_vector(a, b, device=self.device)


OPS_COLQWEN3_TRAINING_DATA = {
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "docmatix-ir",
    "HotpotQA",
    "FEVER",
    "NQ",
    "MIRACLRetrieval",
    "WebInstructSub",  # MathStackExchange and ScienceStackExchange only
    "MrTyDi",
}

multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

OPS_COLQWEN3_CITATION = """
@misc{ops_colqwen3_4b,
  author       = {OpenSearch-AI},
  title        = {Ops-ColQwen3: State-of-the-Art Multimodal Embedding Model for Visual Document Retrieval},
  year         = {2026},
  url          = {https://huggingface.co/OpenSearch-AI/Ops-ColQwen3-4B},
}"""

ops_colqwen3_4b = ModelMeta(
    loader=OpsColQwen3Wrapper,
    name="OpenSearch-AI/Ops-Colqwen3-4B",
    loader_kwargs=dict(dtype=torch.float16, trust_remote_code=True),
    languages=multilingual_langs,
    revision="4894b7d451ff33981650acc693bb482dbef302d3",
    release_date="2026-01-24",
    modalities=["image", "text"],
    n_parameters=4_800_000_000,
    memory_usage_mb=9206,
    max_tokens=32768,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=OPS_COLQWEN3_TRAINING_DATA,
    citation=OPS_COLQWEN3_CITATION,
)
