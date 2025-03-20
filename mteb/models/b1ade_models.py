from __future__ import annotations

from mteb.model_meta import ModelMeta, sentence_transformers_loader

b1ade_training_data = {
    # We are in teh process of submitting a paper outlining our process of creating b1ade using model merging and knowledge distillation.
    # Similar to mixedbread models, we do not train on any data (except the MSMarco training split) of MTEB.
    "MSMARCO": [],
}

b1ade_embed = ModelMeta(
    loader=sentence_transformers_loader,
    name="w601sxs/b1ade-embed",
    languages=["eng-Latn"],
    revision="3bdac13927fdc888b903db93b2ffdbd90b295a69",
    open_weights=True,
    release_date="2025-03-10",
    n_parameters=335_000_000,
    memory_usage_mb=1278,
    embed_dim=1024,
    license="mit",
    max_tokens=4096,
    reference="https://huggingface.co/w601sxs/b1ade-embed",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=b1ade_training_data,
    adapted_from="BAAI/bge-large-en-v1.5",
    # Also (based on model merging)
    # bert-large-uncased
    # WhereIsAI/UAE-Large-V1
    # BAAI/bge-large-en-v1.5
    # mixedbread-ai/mxbai-embed-large-v1
    # avsolatorio/GIST-large-Embedding-v0
)
