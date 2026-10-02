from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

import mteb.models.sentence_transformer_wrapper as st_wrapper
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

LLAMA_NEMORETRIEVER_CITATION = """@misc{xu2025llamanemoretrievercolembedtopperforming,
      title={Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model},
      author={Mengyao Xu and Gabriel Moreira and Ronay Ak and Radek Osmulski and Yauhen Babakhin and Zhiding Yu and Benedikt Schifferer and Even Oldridge},
      year={2025},
      eprint={2507.05513},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05513}
}"""

NEMOTRON_COLEMBED_CITATION_V2 = """
@misc{moreira2026nemotroncolembedv2topperforming,
    title={Nemotron ColEmbed V2: Top-Performing Late Interaction embedding models for Visual Document Retrieval},
    author={Gabriel de Souza P. Moreira and Ronay Ak and Mengyao Xu and Oliver Holworthy and Benedikt Schifferer and Zhiding Yu and Yauhen Babakhin and Radek Osmulski and Jiarui Cai and Ryan Chesler and Bo Liu and Even Oldridge},
    year={2026},
    eprint={2602.03992},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    url={https://arxiv.org/abs/2602.03992},
}"""


class NemotronColEmbedVL(AbsEncoder):
    """Encoder for the NemotronColEmbedVL family of models."""

    def __init__(
        self,
        model_name_or_path: str,
        revision: str,
        trust_remote_code: bool,
        device_map: str = "cuda",
        torch_dtype: OutputDType | torch.dtype = OutputDType.BF16,
        attn_implementation: str = "flash_attention_2",
        **kwargs: Any,
    ):
        if isinstance(torch_dtype, OutputDType):
            torch_dtype = torch_dtype.get_dtype()

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()

    def get_text_embeddings(
        self, texts: DataLoader[BatchedInput], batch_size: int = 32, **kwargs: Any
    ) -> Array:
        return self.model.forward_queries(texts, batch_size=batch_size)

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        import torchvision.transforms.functional as F
        from PIL import Image
        from torch.utils.data import DataLoader

        all_images = []
        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        for batch in iterator:
            for image in batch["image"]:
                pil_img = (
                    image
                    if isinstance(image, Image.Image)
                    else F.to_pil_image(image.to("cpu"))
                )
                all_images.append(pil_img)

        return self.model.forward_images(all_images, batch_size=batch_size)

    def similarity(self, a: Array, b: Array) -> Array:
        return self.model.get_scores(a, b)

    def get_fused_embeddings(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
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
            raise NotImplementedError(
                "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
            )
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError


TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "HotpotQA",
    "MIRACLRetrieval",
    "NQ",
    "StackExchangeClustering",
    "SQuAD",
    "WebInstructSub",
    "docmatix-ir",
    "VDRMultilingualRetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "wiki-ss-nq",
}


TRAINING_DATA_v2 = {
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "docmatix-ir",
    "VDRMultilingualRetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "wiki-ss-nq",
}

llama_nemoretriever_colembed_1b_v1 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/llama-nemoretriever-colembed-1b-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6eade800103413033f260bb55b49fe039fd28a6e",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=2_418_000_000,
    n_embedding_parameters=262688768,
    memory_usage_mb=4610,
    max_tokens=8192,
    embed_dim=2048,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
    extra_requirements_groups=["llama-nemotron-colembed-vl"],
)

llama_nemoretriever_colembed_3b_v1 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/llama-nemoretriever-colembed-3b-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="4194bdd2cd2871f220ddba6273ce173ef1217a1e",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=4_407_000_000,
    n_embedding_parameters=394033152,
    memory_usage_mb=8403,
    max_tokens=8192,
    embed_dim=3072,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
    extra_requirements_groups=["llama-nemotron-colembed-vl"],
)

llama_nemotron_colembed_vl_3b_v2 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/llama-nemotron-colembed-vl-3b-v2",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="680b47b199f99bc0ec2f4e90ffa583ec0c2e452c",
    release_date="2026-01-21",
    modalities=["image", "text"],
    n_parameters=4_407_000_000,
    n_embedding_parameters=394033152,
    memory_usage_mb=8403,
    max_tokens=8192,
    embed_dim=3072,
    license="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=NEMOTRON_COLEMBED_CITATION_V2,
    extra_requirements_groups=["llama-nemotron-colembed-vl"],
)


nemotron_colembed_vl_4b_v2 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/nemotron-colembed-vl-4b-v2",
    revision="0ed152d91f8ad4c5d48296b51c220f686641a398",
    languages=["eng-Latn"],
    release_date="2026-01-07",
    modalities=["image", "text"],
    n_parameters=4_800_000_000,
    n_embedding_parameters=388956160,
    memory_usage_mb=9206,
    max_tokens=262144,
    embed_dim=2560,
    license="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2#training-dataset",
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA_v2,
    citation=NEMOTRON_COLEMBED_CITATION_V2,
    model_type=["late-interaction"],
    extra_requirements_groups=["nemotron-colembed-vl-v2"],
)


nemotron_colembed_vl_8b_v2 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/nemotron-colembed-vl-8b-v2",
    revision="34b640612f311ed05a6c7c62c6564847ed555f5f",
    languages=["eng-Latn"],
    release_date="2026-01-07",
    modalities=["image", "text"],
    n_parameters=8_700_000_000,
    n_embedding_parameters=622329856,
    memory_usage_mb=16722,
    max_tokens=262144,
    embed_dim=4096,
    license="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2#training-dataset",
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA_v2,
    citation=NEMOTRON_COLEMBED_CITATION_V2,
    model_type=["late-interaction"],
    extra_requirements_groups=["nemotron-colembed-vl-v2"],
)


class LlamaNemotronEmbedVL(AbsEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        revision: str,
        trust_remote_code: bool,
        extra_name: str = "llama-nemotron-embed-vl-1b-v2",
        device_map: str = "cuda",
        torch_dtype: OutputDType | torch.dtype = OutputDType.BF16,
        attn_implementation: str = "flash_attention_2",
        use_image_modality: bool = True,
        use_text_modality: bool = True,
        **kwargs: Any,
    ):
        if isinstance(torch_dtype, OutputDType):
            torch_dtype = torch_dtype.get_dtype()

        self.use_image_modality = use_image_modality
        self.use_text_modality = use_text_modality
        if not self.use_image_modality and not self.use_text_modality:
            raise ValueError(
                "At least one of use_image_modality or use_text_modality must be True"
            )

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()

        # Sets the number of tiles the image can be split into
        self.model.processor.max_input_tiles = 4

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
        import torch
        from torch.nn.functional import normalize

        with torch.inference_mode():
            embeddings_list = []
            for batch in tqdm(
                inputs,
                desc=f"Extracting {prompt_type} embeddings...",
                disable=not show_progress_bar,
            ):
                if prompt_type == PromptType.query and "text" in batch:
                    embeddings = self.model.encode_queries(batch["text"])
                else:
                    if not self.use_image_modality and "image" in batch:
                        del batch["image"]
                    if not self.use_text_modality and "text" in batch:
                        del batch["text"]

                    if "image" in batch and "text" in batch:
                        embeddings = self.model.encode_documents(
                            images=batch["image"], texts=batch["text"]
                        )
                    elif "image" in batch:
                        embeddings = self.model.encode_documents(images=batch["image"])
                    elif "text" in batch:
                        embeddings = self.model.encode_documents(texts=batch["text"])
                    else:
                        raise ValueError(
                            f"Could not find 'image' or 'text' in batch: {batch}"
                        )

                embeddings = normalize(embeddings, dim=-1)
                if torch.sum(embeddings).float().item() in {0.0, float("inf")}:
                    raise ValueError("Embeddings sum is invalid (0.0 or inf)")
                embeddings_list.append(embeddings)

            concatenated_embeddings = torch.vstack(embeddings_list)
            return concatenated_embeddings


LLAMA_NEMOTRON_VL_1B_V2_LANGUAGES = [
    "eng-Latn",
    "ara-Arab",
    "ben-Beng",
    "zho-Hans",
    "ces-Latn",
    "dan-Latn",
    "nld-Latn",
    "fin-Latn",
    "fra-Latn",
    "deu-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hun-Latn",
    "ind-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "nor-Latn",
    "fas-Arab",
    "pol-Latn",
    "por-Latn",
    "rus-Cyrl",
    "spa-Latn",
    "swe-Latn",
    "tha-Thai",
    "tur-Latn",
]

LLAMA_NEMOTRON_VL_1B_V2_CITATION = """@inproceedings{moreira2025_nvretriever,
  author = {Moreira, Gabriel de Souza P. and Osmulski, Radek and Xu, Mengyao and Ak, Ronay and Schifferer, Benedikt and Oldridge, Even},
  title = {Improving Text Embedding Models with Positive-aware Hard-negative Mining},
  year = {2025},
  isbn = {9798400720406},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746252.3761254},
  doi = {10.1145/3746252.3761254},
  pages = {2169–2178},
  numpages = {10},
  keywords = {contrastive learning, distillation, embedding models, hard-negative mining, rag, text retrieval, transformers},
  location = {Seoul, Republic of Korea},
  series = {CIKM '25},
}"""

TRAINING_DATA_EMBED_VL_1B_V2 = {
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "docmatix-ir",
    "wiki-ss-nq",
    "Cauldron (AI2D, OCRVQA, Websight)",
    "VDRMultilingualRetrieval",
    "HotpotQA",
    "MIRACLRetrieval",
    "NQ",
    "StackExchangeClustering",
    "SQuAD",
    "MultiLongDocRetrieval",
    "MLQARetrieval",
    "Tiger Math/Stack",
}

llama_nemotron_embed_vl_1b_v2 = ModelMeta(
    loader=LlamaNemotronEmbedVL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="nvidia/llama-nemotron-embed-vl-1b-v2",
    languages=LLAMA_NEMOTRON_VL_1B_V2_LANGUAGES,
    revision="859e1f2dac29c56c37a5279cf55f53f3e74efc6b",
    release_date="2026-01-06",
    modalities=["image", "text"],
    n_parameters=1_678_252_480,
    n_embedding_parameters=262_688_768,
    memory_usage_mb=6402,
    max_tokens=10240,
    embed_dim=2048,
    license="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    open_weights=True,
    public_training_code="https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/retrieval/bi_encoder/nemotron_vl_1b",
    public_training_data="https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2#training-dataset",
    framework=["PyTorch"],
    reference="https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=TRAINING_DATA_EMBED_VL_1B_V2,
    citation=LLAMA_NEMOTRON_VL_1B_V2_CITATION,
    extra_requirements_groups=["llama-nemotron-colembed-vl"],
)


llama_nemotron_rerank_vl_1b_v2 = ModelMeta(
    loader=st_wrapper.CrossEncoderWrapper,
    loader_kwargs={"trust_remote_code": True},
    name="nvidia/llama-nemotron-rerank-vl-1b-v2",
    revision="b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a",
    release_date="2025-12-18",
    languages=LLAMA_NEMOTRON_VL_1B_V2_LANGUAGES,
    n_parameters=1_678_256_576,
    n_embedding_parameters=262_690_816,
    memory_usage_mb=3201,
    max_tokens=10240,
    embed_dim=None,
    license="https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2#training-dataset",
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # Training inherited from the text reranker backbone.
        "NQ",
        "HotpotQA",
        "MIRACLRetrieval",
        "MLQARetrieval",
        "MultiLongDocRetrieval",
        # Vision-language training.
        "VidoreDocVQARetrieval",
        "VidoreTatdqaRetrieval",
        "VidoreArxivQARetrieval",
        "VidoreInfoVQARetrieval",
    },
    modalities=["image", "text"],
    model_type=["cross-encoder"],
    citation=LLAMA_NEMOTRON_VL_1B_V2_CITATION,
)
