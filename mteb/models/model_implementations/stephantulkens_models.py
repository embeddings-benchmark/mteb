from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[call-arg]
    loader_kwargs={},
    name="stephantulkens/NIFE-gte-modernbert-base",
    revision="9b85ba93d5a7729db11dfea90d20a5f392a88747",
    release_date="2025-10-30",
    languages=["eng-Latn"],  # assumed from the original model
    n_parameters=76802304,  # TODO: what do we do for routers? Both models I assume? - this is for the query router / student model
    n_active_parameters_override=None,  # TODO: not sure how to count this for routers - WDYT?
    n_embedding_parameters=76802304,  # this is for the query router / student model
    memory_usage_mb=293.0,
    max_tokens=8192.0,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code=None,  # no reproducible training code, but likely uses https://pypi.org/project/pynife/
    public_training_data=None,  # its a distilaltion of a model
    framework=["PyTorch", "safetensors", "Sentence Transformers"],
    reference="https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,  # assumed
    training_datasets=set(),
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],  # TODO: is router a model type?
    citation="""@software{Tulkens2025pyNIFE,
  author       = {St\'{e}phan Tulkens},
  title        = {pyNIFE: nearly inference free embeddings in python},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17512919},
  url          = {https://github.com/stephantulkens/pynife},
  license      = {MIT},
}""",
    contacts=None,
    adapted_from="Alibaba-NLP/gte-modernbert-base",
)

ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[call-arg]
    loader_kwargs={},
    name="stephantulkens/NIFE-mxbai-embed-large-v1",
    revision="970a0151f3faeddbab5c68be38d52bbc5ea719ab",
    release_date="2025-11-03",
    languages=["eng-Latn"],  # assumed from the original model
    n_parameters=110553088,  # TODO: what do we do for routers? Both models I assume? - this is for the query router / student model
    n_active_parameters_override=None,  # TODO: not sure how to count this for routers - WDYT?
    n_embedding_parameters=110553088,  # this is for the query router / student model
    memory_usage_mb=422.0,
    max_tokens=512,  # assumed from the original model
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=None,  # no reproducible training code, but likely uses https://pypi.org/project/pynife/
    public_training_data=None,  # its a distilaltion of a model
    framework=["PyTorch", "safetensors", "Sentence Transformers"],
    reference="https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,  # assumed
    training_datasets=set(),
    adapted_from="mixedbread-ai/mxbai-embed-large-v1",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],  # TODO: is router a model type?
    citation=None,
    contacts=None,
)
