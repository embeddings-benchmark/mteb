from __future__ import annotations

from mteb.model_meta import ModelMeta

arabic_triplet_matryoshka = ModelMeta(
    name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    languages=["ara-Arab"],
    open_weights=True,
    revision="ed357f222f0b6ea6670d2c9b5a1cb93950d34200",
    release_date="2024-07-28",
    n_parameters=135_000_000,
    memory_usage_mb=516,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=768,
    reference="https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    adapted_from="aubmindlab/bert-base-arabertv02",
    public_training_data="akhooli/arabic-triplets-1m-curated-sims-len",
    training_datasets={
        #  "akhooli/arabic-triplets-1m-curated-sims-len"
    },
)
