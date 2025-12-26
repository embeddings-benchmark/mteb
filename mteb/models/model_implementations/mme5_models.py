from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerMultimodalEncoderWrapper,
)

mme5_mllama = ModelMeta(
    loader=SentenceTransformerMultimodalEncoderWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="intfloat/mmE5-mllama-11b-instruct",
    model_type=["dense"],
    revision="cbb328b9bf9ff5362c852c3166931903226d46f1",
    release_date="2025-02-12",
    languages=["eng-Latn"],
    n_parameters=10_600_000_000,  # 10.6B
    memory_usage_mb=20300,
    max_tokens=128_000,
    embed_dim=4096,
    license="mit",
    modalities=["text", "image"],
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/intfloat/mmE5-MMEB-hardneg, https://huggingface.co/datasets/intfloat/mmE5-synthetic",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/intfloat/mmE5-mllama-11b-instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=set(),  # synthetic training dataset
    adapted_from="meta-llama/Llama-3.2-11B-Vision",
    citation="""
@article{chen2025mmE5,
  title={mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data},
  author={Chen, Haonan and Wang, Liang and Yang, Nan and Zhu, Yutao and Zhao, Ziliang and Wei, Furu and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2502.08468},
  year={2025}
}
""",
)
