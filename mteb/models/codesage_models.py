from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

codesage_languages = [
    "python-Code",
    "javascript-Code",
    "go-Code",
    "ruby-Code",
    "java-Code",
    "php-Code",
]

codesage_large = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-large-v2",
        revision="6e5d6dc15db3e310c37c6dbac072409f95ffa5c5",
    ),
    name="codesage/codesage-large-v2",
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
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)

codesage_base = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-base-v2",
        revision="92eac4f44c8674638f039f1b0d8280f2539cb4c7",
    ),
    name="codesage/codesage-base-v2",
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
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)

codesage_small = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-small-v2",
        revision="4844c2f24b25e181aa43ca058cc73dd2622565c1",
    ),
    name="codesage/codesage-small-v2",
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
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)
