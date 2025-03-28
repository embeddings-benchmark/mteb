from __future__ import annotations

from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from functools import partial
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download


class YuanWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer(
            "IEITYuan/Yuan-embedding-1.0", trust_remote_code=True
        )

        model_path = snapshot_download(
            repo_id="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
            local_dir="./Ops_MoA_Yuan/",
        )
        from Ops_MoA_Yuan.modeling_adaptor import MixtureOfAdaptors

        adaptor = MixtureOfAdaptors(5, 1792)

        adaptor.load_state_dict(torch.load(f"Ops_MoA_Yuan/yuan-adaptors.pth"))

        self.model.add_module("adaptor", adaptor)
        self.output_dim = 1536

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.output_dim]


class ConanWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer(
            "TencentBAC/Conan-embedding-v1", trust_remote_code=True
        )

        model_path = snapshot_download(
            repo_id="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
            local_dir="./Ops_MoA_Conan",
        )
        from Ops_MoA_Conan.modeling_adaptor import MixtureOfAdaptors

        adaptor = MixtureOfAdaptors(5, 1792)

        adaptor.load_state_dict(torch.load(f"Ops_MoA_Conan/conan-adaptors.pth"))

        self.model.add_module("adaptor", adaptor)
        self.output_dim = 1536

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.output_dim]


ops_moa_conan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    revision="cd42de6d61c103047b7bcd780ef0dbaa9a9d0472",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=partial(
        ConanWrapper,
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
    superseded_by=None,
)

ops_moa_yuan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    revision="09b8857bbd74f189d9bc45bf59adaf34f9378e17",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=partial(
        YuanWrapper,
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
    superseded_by=None,
)
