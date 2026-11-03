from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)
from mteb.types import OutputDType

mme5_mllama = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="intfloat/mmE5-mllama-11b-instruct",
    model_type=["dense"],
    revision="cbb328b9bf9ff5362c852c3166931903226d46f1",
    release_date="2025-02-12",
    languages=["eng-Latn"],
    n_parameters=10_600_000_000,
    n_embedding_parameters=525336576,
    memory_usage_mb=20300,
    max_tokens=128_000,
    embed_dim=4096,
    license="mit",
    modalities=["text", "image"],
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/intfloat/mmE5-MMEB-hardneg, https://huggingface.co/datasets/intfloat/mmE5-synthetic",
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/intfloat/mmE5-mllama-11b-instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=set(),  # synthetic training dataset
    adapted_from="meta-llama/Llama-3.2-11B-Vision",
    citation="""
@inproceedings{chen2025mmE5,
  address = {Vienna, Austria},
  author = {Chen, Haonan and Wang, Liang and Yang, Nan and Zhu, Yutao and Zhao, Ziliang and Wei, Furu and Dou, Zhicheng},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025},
  doi = {10.18653/v1/2025.findings-acl.433},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-256-5},
  month = {jul},
  pages = {8254--8275},
  publisher = {Association for Computational Linguistics},
  title = {mm{E}5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data},
  url = {https://aclanthology.org/2025.findings-acl.433/},
  year = {2025},
}
""",
    output_dtypes=OutputDType.BF16,
)
