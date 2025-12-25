from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

CODESAGE_CITATION = """@inproceedings{
    zhang2024code,
    title={{CODE} {REPRESENTATION} {LEARNING} {AT} {SCALE}},
    author={Dejiao Zhang and Wasi Uddin Ahmad and Ming Tan and Hantian Ding and Ramesh Nallapati and Dan Roth and Xiaofei Ma and Bing Xiang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=vfzRRjumpX}
}"""

codesage_languages = [
    "python-Code",
    "javascript-Code",
    "go-Code",
    "ruby-Code",
    "java-Code",
    "php-Code",
]

codesage_large = ModelMeta(
    loader=sentence_transformers_loader,
    name="codesage/codesage-large-v2",
    model_type=["dense"],
    languages=codesage_languages,
    revision="6e5d6dc15db3e310c37c6dbac072409f95ffa5c5",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=1_300_000_000,
    memory_usage_mb=4959,
    max_tokens=2048,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-large-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
    },
    citation=CODESAGE_CITATION,
)

codesage_base = ModelMeta(
    loader=sentence_transformers_loader,
    name="codesage/codesage-base-v2",
    model_type=["dense"],
    languages=codesage_languages,
    revision="92eac4f44c8674638f039f1b0d8280f2539cb4c7",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=356_000_000,
    memory_usage_mb=1358,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-base-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
    },
    citation=CODESAGE_CITATION,
)

codesage_small = ModelMeta(
    loader=sentence_transformers_loader,
    name="codesage/codesage-small-v2",
    model_type=["dense"],
    languages=codesage_languages,
    revision="4844c2f24b25e181aa43ca058cc73dd2622565c1",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=130_000_000,
    memory_usage_mb=496,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-small-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
    },
    citation=CODESAGE_CITATION,
)
