import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta


class OPSWrapper(AbsEncoder):
    def __init__(self, model_name: str, revision: str):
        from sentence_transformers import SentenceTransformer

        super().__init__()

        self.model = SentenceTransformer(
            model_name, revision=revision, trust_remote_code=True
        )
        self.output_dim = 1536

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.output_dim]


ops_moa_conan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    revision="46dcd58753f3daa920c66f89e47086a534089350",
    release_date="2025-03-26",
    languages=["zho-Hans"],
    loader=OPSWrapper,
    n_parameters=int(343 * 1e6),
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "T2Retrieval",
        "MMarcoRetrieval",
        "DuRetrieval",
        "CovidRetrieval",
        "CmedqaRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "VideoRetrieval",
    },
    superseded_by=None,
)

ops_moa_yuan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    revision="23712d0766417b0eb88a2513c6e212a58b543268",
    release_date="2025-03-26",
    languages=["zho-Hans"],
    loader=OPSWrapper,
    n_parameters=int(343 * 1e6),
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "T2Retrieval",
        "MMarcoRetrieval",
        "DuRetrieval",
        "CovidRetrieval",
        "CmedqaRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "VideoRetrieval",
    },
    superseded_by=None,
)
