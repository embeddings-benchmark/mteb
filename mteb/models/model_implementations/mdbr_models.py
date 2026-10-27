from mteb.models.model_implementations.arctic_models import arctic_v1_training_datasets
from mteb.models.model_implementations.mixedbread_ai_models import (
    mixedbread_training_data,
)
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

LEAF_CITATION = """@inproceedings{mdbr_leaf,
  address = {San Diego, California, United States},
  author = {Vujanic, Robin and R{\\"u}ckstie{\\ss}, Thomas},
  booktitle = {Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2026.acl-long.2008},
  editor = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  isbn = {979-8-89176-390-6},
  month = {jul},
  pages = {43362--43383},
  publisher = {Association for Computational Linguistics},
  title = {{LEAF}: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations},
  url = {https://aclanthology.org/2026.acl-long.2008/},
  year = {2026},
}"""

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}

LEAF_TRAINING_DATASETS = {
    "AmazonQA",
    "LoTTE",
    # FineWeb
    # CC-News
    # PubMedQA
    # TriviaQA
}

mdbr_leaf_ir = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="MongoDB/mdbr-leaf-ir",
    model_type=["dense"],
    revision="2e46f5aac796e621d51f678c306a66ede4712ecb",
    release_date="2025-08-27",
    languages=["eng-Latn"],
    open_weights=True,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    n_parameters=22713216,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/MongoDB/mdbr-leaf-ir",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=LEAF_TRAINING_DATASETS | arctic_v1_training_datasets,
    citation=LEAF_CITATION,
)

mdbr_leaf_mt = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="MongoDB/mdbr-leaf-mt",
    model_type=["dense"],
    revision="66c47ba6d753efc208d54412b5af6c744a39a4df",
    release_date="2025-08-27",
    languages=["eng-Latn"],
    open_weights=True,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    n_parameters=22713216,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/MongoDB/mdbr-leaf-mt",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=LEAF_TRAINING_DATASETS | mixedbread_training_data,
    citation=LEAF_CITATION,
)
