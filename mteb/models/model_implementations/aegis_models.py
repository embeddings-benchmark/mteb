from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .bge_models import bge_m3_training_data, bgem3_languages

aegis_embed = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="llm-semantic-router/aegis-embed",
    model_type=["dense"],
    languages=bgem3_languages,
    open_weights=True,
    revision="2a9ae9e597c814c00fd7ccf593eb03dad455ae1e",
    release_date="2025-04-16",
    n_parameters=306_939_648,
    n_embedding_parameters=196_608_000,
    memory_usage_mb=585,
    embed_dim=[768, 512, 256, 128, 64],
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/llm-semantic-router/aegis-embed",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=False,
    adapted_from="llm-semantic-router/mmbert-32k-yarn",
    superseded_by=None,
    training_datasets=bge_m3_training_data,
    public_training_code=None,
    public_training_data=None,
    citation="""@misc{aegis-embed,
  title={aegis-embed: Multilingual Embedding Model with 2D Matryoshka Training},
  author={vLLM Semantic Router Team},
  year={2025},
  url={https://huggingface.co/llm-semantic-router/aegis-embed}
}""",
)
