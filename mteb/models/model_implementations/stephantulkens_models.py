from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

TRAINING_DATA = {
    "MSMARCO",
    "MIRACLRetrieval",
    "LoTTE",
    "TriviaQA",
    "GooAQ",
    "MrTidyRetrieval",
}

CITAITON = """@software{Tulkens2025pyNIFE,
  author       = {St\'{e}phan Tulkens},
  title        = {pyNIFE: nearly inference free embeddings in python},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17512919},
  url          = {https://github.com/stephantul/pynife},
  license      = {MIT},
}"""

nife_gte_modernbert_base_as_router_meta = ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[call-arg]
    loader_kwargs={},
    name="stephantulkens/NIFE-gte-modernbert-base_as_router",
    revision="9b85ba93d5a7729db11dfea90d20a5f392a88747",
    release_date="2025-10-30",
    languages=["eng-Latn"],  # assumed from the original model
    n_parameters=225816576,  # query router + corpus router
    n_active_parameters_override=0,  # active parameters is 0 for the query router
    n_embedding_parameters=76802304 + 38682624,  # query router + corpus router
    memory_usage_mb=861,  # both routers together
    max_tokens=8192.0,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code=None,  # no reproducible training code, but uses https://pypi.org/project/pynife/
    public_training_data="https://huggingface.co/collections/stephantulkens/gte-modernbert-embedpress",  # + its a distilaltion of a model
    framework=["PyTorch", "safetensors", "Sentence Transformers"],
    reference="https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,  # assumed
    training_datasets=TRAINING_DATA,
    superseded_by=None,
    modalities=["text"],
    model_type=["router"],
    citation=CITAITON,
    contacts=["stephantul"],
    adapted_from="Alibaba-NLP/gte-modernbert-base",
)

nife_mxbai_embed_large_v1_as_router_meta = ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[call-arg]
    loader_kwargs={},
    name="stephantulkens/NIFE-mxbai-embed-large-v1_as_router",
    revision="d806c54543eb3456d78cb55567ac1631ab4c72d8",
    release_date="2025-11-03",
    languages=["eng-Latn"],  # assumed from the original model
    n_parameters=445694976,  # corpus + query router
    n_active_parameters_override=0,  # active parameters is 0 for the query router
    n_embedding_parameters=110553088 + 31_254_528,  # query router + corpus router
    memory_usage_mb=1700,  # both routers together
    max_tokens=512,  # corpus router is trained with max length 512, but the query router / student model is static (so infinite)
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=None,  # no reproducible training code, but uses https://pypi.org/project/pynife/
    public_training_data=None,  # its a distilaltion of a model
    framework=["PyTorch", "safetensors", "Sentence Transformers"],
    reference="https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1_as_router",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=TRAINING_DATA,
    adapted_from="mixedbread-ai/mxbai-embed-large-v1",
    superseded_by=None,
    modalities=["text"],
    model_type=["router"],
    citation=CITAITON,
    contacts=["stephantul"],
)
