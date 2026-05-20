from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .bge_models import bge_m3_training_data, bgem3_languages

elephant_embeddings_v1_text_small = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="llm-semantic-router/elephant-embeddings-v1-text-small",
    model_type=["dense"],
    languages=bgem3_languages,
    open_weights=True,
    revision="b91093e77c994fbafad18bf6d1c14f0a7624921e",
    release_date="2025-04-16",
    n_parameters=306_939_648,
    n_embedding_parameters=196_608_000,
    memory_usage_mb=585,
    embed_dim=[768, 512, 256, 128, 64],
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/llm-semantic-router/elephant-embeddings-v1-text-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=False,
    adapted_from="llm-semantic-router/mmbert-32k-yarn",
    superseded_by=None,
    training_datasets=bge_m3_training_data,
    public_training_code=None,
    public_training_data=None,
    citation="""@misc{mmbert-embed-2d-matryoshka,
  title={mmBERT-Embed: Multilingual Embedding Model with 2D Matryoshka Training},
  author={vLLM Semantic Router Team},
  year={2025},
  url={https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka}
}""",
)
