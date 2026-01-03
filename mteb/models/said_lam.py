"""SAID-LAM model metadata for MTEB."""

from mteb import ModelMeta

said_lam_v1 = ModelMeta(
    name="SAID-LAM-v1",
    revision="1.0.0",
    languages=["eng"],
    open_source=False,
    model_type="sentence-embedding",
    max_tokens=32000,  # Free tier: 12K tokens, Licensed tier: 32K tokens
    embedding_dimensions=384,
    release_date="2026-01-01",
    framework=["pytorch"],  # PyTorch with compiled C++ extensions
    org_name="Said-Research",
    description="Linear Attention Memory (LAM) with SAID Crystalline Attention (SCA) - BETA. Achieves perfect recall (100% on LongEmbed needle-in-haystack tasks) via deterministic attention with 0.0% signal loss. First model to demonstrate true crystalline memory across all context lengths. Free tier: 12K tokens, Licensed tier: 32K tokens. Optimized with compiled C++ extensions.",
    reference="https://saidhome.ai",
    architecture="LAM",
    license="proprietary",
)

