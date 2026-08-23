"""Text embedding models for Mauritian Creole (Kreol Morisien, mfe)."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

morisien_embed = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="Singaraj/morisien-embed",
    model_type=["dense"],
    languages=["mfe-Latn", "eng-Latn", "fra-Latn"],
    open_weights=True,
    revision="3187caaa709a8ea65a27342bddf87429e29d35f6",
    release_date="2026-08-09",
    n_parameters=278_043_648,
    n_embedding_parameters=192_001_536,
    memory_usage_mb=1061,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/Singaraj/morisien-embed",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="intfloat/multilingual-e5-base",
    training_datasets={"MorisienMTBitextMining"},
    public_training_code="https://github.com/LK-maker-007/morisien-embed",
    public_training_data=None,
    citation="""@misc{b2026morisienembed,
  title={morisien-embed: a dedicated text embedding model for Mauritian Creole},
  author={B, Singaraj},
  year={2026},
  url={https://huggingface.co/Singaraj/morisien-embed},
}""",
)
