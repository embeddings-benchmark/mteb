from mteb.models.model_implementations.e5_models import ME5_TRAINING_DATA, model_prompts
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

INDIC_E5_LANGUAGES = [
    "hin-Deva",
    "ben-Beng",
    "tam-Taml",
    "tel-Telu",
    "mal-Mlym",
    "mar-Deva",
    "guj-Gujr",
    "kan-Knda",
    "pan-Guru",
    "ory-Orya",
    "asm-Beng",
    "urd-Arab",
    "npi-Deva",
    "eng-Latn",
    # romanized (Latin-script) Indic text
    "hin-Latn",
    "ben-Latn",
    "tam-Latn",
    "tel-Latn",
    "mal-Latn",
    "mar-Latn",
    "guj-Latn",
    "kan-Latn",
    "pan-Latn",
    "ory-Latn",
    "asm-Latn",
    "urd-Latn",
    "npi-Latn",
]

indic_e5_small = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="cidcodes/indic-e5-small",
    model_type=["dense"],
    languages=INDIC_E5_LANGUAGES,
    open_weights=True,
    revision="08d044b8792a0d345fc3d7911036edbe251e5425",
    release_date="2026-11-27",
    n_parameters=46_805_376,
    n_embedding_parameters=25_165_824,
    memory_usage_mb=179,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/cidcodes/indic-e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/codewithcid/indic-embed",
    public_training_data="https://huggingface.co/datasets/ai4bharat/sangraha",
    # vocabulary-trimmed multilingual-e5-small, distilled on Sangraha (not an MTEB dataset)
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="intfloat/multilingual-e5-small",
)
