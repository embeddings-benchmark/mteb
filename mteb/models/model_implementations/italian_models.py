from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

bge_m3_italian = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="albertobarnabo/bge-m3-italian",
    model_type=["dense"],
    languages=["ita-Latn"],
    open_weights=True,
    revision="f22ed34934c792dee1a399e487d15d071754666f",
    release_date="2026-09-13",
    n_parameters=567_754_752,
    n_embedding_parameters=256_002_048,
    memory_usage_mb=2167,
    embed_dim=1024,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/albertobarnabo/bge-m3-italian",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="BAAI/bge-m3",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/unicamp-dl/mmarco",
    # Fine-tuned on the Italian split of mMARCO (train queries and passages), so
    # the mteb task built on that data is listed here; the bge-m3 base data is
    # inherited through adapted_from.
    training_datasets={"MMarcoRetrievalMultilingual"},
)
